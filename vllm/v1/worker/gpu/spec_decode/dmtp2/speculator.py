# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-MTP2 speculator: frozen-MTP first token + DFlash2 parallel continuation.

Each propose() runs two draft passes:

1. **MTP pass** (eagle-style draft prefill on the target batch): the shifted
   target tokens paired with the target's last hidden states run through the
   frozen MTP block, keeping its KV cache in sync; each request's last valid
   row yields h_MTP, whose greedy argmax is T1.
2. **DFlash2 pass**: the usual parallel block pass with the dual-anchor layout
   ``[A, T1, MASK x (B-2)]`` -- slot 1 carries T1 with the fusion residual
   added to its embedding, slots 2+ are drafted, and the candidate-selector
   walk is seeded with T1 as the first predecessor.

Current limitations (eager-first bring-up):
- CUDA graphs are disabled (the MTP pass and T1 injection are not captured).
- Greedy draft sampling only (``draft_sample_method="probabilistic"`` and
  adaptive verification are rejected at startup).
- No ubatching / prefill context parallelism.
"""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, set_forward_context
from vllm.logger import init_logger
from vllm.v1.worker.gpu.dp_utils import DPSyncState
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.dflash.utils import load_dflash_model
from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
    CandidateSampler,
    DFlash2Speculator,
)
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar.speculator import (
    prepare_prefill_inputs,
)

logger = init_logger(__name__)


class DMTP2Speculator(DFlash2Speculator):
    _speculator_name = "DMTP2"

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        if self.num_speculative_steps < 2:
            raise ValueError(
                "D-MTP2 needs num_speculative_tokens >= 2 (T1 + at least one "
                f"DFlash2 slot); got {self.num_speculative_steps}."
            )
        if self.use_draft_logits:
            raise NotImplementedError(
                "D-MTP2 only supports draft_sample_method='greedy' for now."
            )
        if self.enable_adaptive_verification:
            raise NotImplementedError(
                "D-MTP2 does not support adaptive verification yet."
            )
        if vllm_config.parallel_config.use_ubatching:
            raise NotImplementedError(
                "D-MTP2 cannot reuse the target attention metadata with "
                "ubatching enabled."
            )
        if self.dp_size > 1:
            raise NotImplementedError(
                "D-MTP2's MTP pass does not synchronize token counts across "
                "data-parallel ranks yet (MoE expert dispatch would hang)."
            )
        if vllm_config.parallel_config.enable_eplb:
            raise NotImplementedError(
                "D-MTP2 does not prepare EPLB state for the frozen MTP MoE "
                "forward yet."
            )

        # Dual-anchor layout: slot 0 is the bonus anchor, slot 1 is T1, and
        # only the remaining block slots are drafted by the DFlash2 pass.
        self.first_draft_offset = 2
        self.num_sample_steps = self.num_speculative_steps - 1
        self.sample_col = torch.arange(
            self.num_sample_steps, dtype=torch.int32, device=device
        ).repeat(self.max_num_reqs)
        self.candidate_sampler = CandidateSampler(
            self.max_num_reqs, self.num_sample_steps, self.top_k, device
        )

        # The selector walk writes its tokens here; draft_tokens[:, 0] is T1.
        self.walk_tokens = torch.zeros(
            self.max_num_reqs,
            self.num_sample_steps,
            dtype=torch.int64,
            device=device,
        )
        self.t1_tokens = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self.t1_hidden = torch.zeros(
            self.max_num_reqs, self.hidden_size, dtype=self.dtype, device=device
        )
        # MTP-pass bookkeeping (eagle-style prefill on the target batch).
        self.mtp_last_token_indices = torch.zeros(
            self.max_num_reqs, dtype=torch.int64, device=device
        )
        self._mtp_step_scratch = torch.zeros((), dtype=torch.int64, device=device)
        self._fusion_rows = (
            torch.arange(self.max_num_reqs, dtype=torch.int64, device=device)
            * self.num_query_per_req
            + 1
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        # Eager-only bring-up: the MTP pass and the T1 injection between the
        # two draft passes are not captured yet.
        if cudagraph_mode.decode_mode() != CUDAGraphMode.NONE:
            logger.warning(
                "D-MTP2 drafting does not support CUDA graphs yet; running "
                "the draft eagerly."
            )
        super().init_cudagraph_manager(CUDAGraphMode.NONE)

    def load_draft_model(
        self,
        target_model: nn.Module,
        target_attn_layer_names: set[str],
    ) -> nn.Module:
        model = load_dflash_model(target_model, self.vllm_config)
        # The MTP block embeds the shifted target tokens with the verifier's
        # embedding, which the draft already shares with the target.
        model.mtp.embed_tokens = model.model.embed_tokens
        return model

    @torch.inference_mode()
    def _propose_t1(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
    ) -> None:
        """Run the frozen MTP block over the target batch: keeps its KV cache
        in sync and proposes T1 (+ h_MTP) for every request."""
        num_reqs = input_batch.num_reqs
        num_tokens_padded = input_batch.num_tokens_after_padding

        # Shift the target tokens into the draft input buffers (row t carries
        # x_{t+1}) and record each request's last valid row. Runs before
        # prepare_dflash_inputs, which overwrites the same buffers.
        prepare_prefill_inputs(
            self.mtp_last_token_indices,
            self._mtp_step_scratch,
            self.input_buffers,
            input_batch,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            self.max_num_reqs,
        )

        mtp_attn_metadata, mtp_slot_mappings = self._build_drafter_attn_metadata(
            input_batch, attn_metadata, slot_mappings, CUDAGraphMode.NONE
        )
        with set_forward_context(
            mtp_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
            slot_mapping=mtp_slot_mappings,
            batch_descriptor=BatchDescriptor(num_tokens=num_tokens_padded),
        ):
            mtp_hidden = self.model.mtp_forward(
                input_ids=self.input_buffers.input_ids[:num_tokens_padded],
                positions=self.input_buffers.positions[:num_tokens_padded],
                hidden_states=last_hidden_states[:num_tokens_padded],
            )

        rows = self.mtp_last_token_indices[:num_reqs]
        h_mtp = mtp_hidden[rows]
        self.t1_hidden[:num_reqs] = h_mtp
        t1_logits = self.model.compute_t1_logits(h_mtp)
        self.t1_tokens[:num_reqs] = t1_logits.argmax(dim=-1)

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: dict[str, Any],
        slot_mappings: dict[str, torch.Tensor],
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        if dummy_run:
            # Profiling/dummy paths skip the real MTP pass; the fusion then adds
            # a zero-normed residual to a mask-token slot, which is shape-exact.
            self.t1_tokens[:num_reqs] = self.parallel_drafting_token_id
            self.t1_hidden[:num_reqs].zero_()
            if skip_attn_for_dummy_run:
                # Memory profiling: run the MTP forward without attention
                # metadata so its activations count toward the peak.
                num_tokens_padded = input_batch.num_tokens_after_padding
                with set_forward_context(
                    None, self.vllm_config, num_tokens=num_tokens_padded
                ):
                    self.model.mtp_forward(
                        input_ids=self.input_buffers.input_ids[:num_tokens_padded],
                        positions=self.input_buffers.positions[:num_tokens_padded],
                        hidden_states=last_hidden_states[:num_tokens_padded],
                    )
        else:
            # Runs even when no drafts are needed this step: the MTP KV cache
            # must stay in sync with the accepted tokens.
            self._propose_t1(
                input_batch,
                attn_metadata,
                slot_mappings,
                last_hidden_states,
                num_sampled,
                num_rejected,
                last_sampled,
                next_prefill_tokens,
            )
        return super().propose(
            input_batch,
            attn_metadata,
            slot_mappings,
            last_hidden_states,
            aux_hidden_states,
            num_sampled,
            num_rejected,
            last_sampled,
            next_prefill_tokens,
            temperature,
            seeds,
            dp_sync_state=dp_sync_state,
            dummy_run=dummy_run,
            skip_attn_for_dummy_run=skip_attn_for_dummy_run,
            mm_inputs=mm_inputs,
            is_profile=is_profile,
            num_speculative_tokens=num_speculative_tokens,
        )

    def _generate_draft(
        self,
        num_reqs: int,
        num_tokens_padded: int,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> None:
        num_query_tokens = num_reqs * self.num_query_per_req
        input_ids = self.input_buffers.input_ids
        t1 = self.t1_tokens[:num_reqs]
        # Inject T1 into slot 1 of every block (prepare_dflash_inputs wrote the
        # mask token there) and add the fusion residual to its embedding.
        input_ids[1:num_query_tokens:self.num_query_per_req] = t1
        inputs_embeds = self.model.embed_input_ids(input_ids[:num_tokens_padded])
        fusion_rows = self._fusion_rows[:num_reqs]
        inputs_embeds[fusion_rows] = self.model.fuse_t1_embedding(
            inputs_embeds[fusion_rows], self.t1_hidden[:num_reqs]
        ).to(inputs_embeds.dtype)

        last_hidden_states = self._run_model_with_embeds(
            num_tokens_padded,
            inputs_embeds,
            attn_metadata,
            slot_mappings,
            num_tokens_across_dp,
            cudagraph_runtime_mode,
        )

        num_sample = num_reqs * self.num_sample_steps
        hidden_states = last_hidden_states[self.sample_indices[:num_sample]].view(
            num_reqs, self.num_sample_steps, -1
        )
        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden_states.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(
            num_reqs, self.num_sample_steps, self.top_k
        )
        unary_logits = unary_logits.view_as(candidate_ids)
        # The first walked slot's predecessor is T1 (not the anchor).
        scores = self.model.model.candidate_selector(
            candidate_ids, unary_logits, hidden_states, t1
        )
        self.candidate_sampler.sample(
            candidate_ids,
            scores,
            num_reqs,
            self.sample_pos,
            self.sample_idx_mapping,
            self.temperature,
            self.seeds,
            self.walk_tokens,
            None,  # draft_logits: greedy-only for now
            self.use_fp64_gumbel,
        )
        self.draft_tokens[:num_reqs, 0] = t1
        self.draft_tokens[:num_reqs, 1:] = self.walk_tokens[:num_reqs]

    @torch.inference_mode()
    def _run_model_with_embeds(
        self,
        num_tokens: int,
        inputs_embeds: torch.Tensor,
        attn_metadata: dict[str, Any] | None,
        slot_mappings: dict[str, torch.Tensor] | None,
        num_tokens_across_dp: torch.Tensor | None,
        cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    ) -> torch.Tensor:
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens)
        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            num_tokens_across_dp=num_tokens_across_dp,
            slot_mapping=slot_mappings,
            batch_descriptor=batch_descriptor,
            is_padding=self.input_buffers.is_padding[:num_tokens],
        ):
            return self.model(
                input_ids=self.input_buffers.input_ids[:num_tokens],
                positions=self.input_buffers.positions[:num_tokens],
                inputs_embeds=inputs_embeds,
            )
