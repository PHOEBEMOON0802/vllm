# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""D-MTP2 draft model: DFlash2 backbone + frozen-MTP first token.

D-MTP2 (https://sylvan820.github.io/D-MTP/) keeps the DFlash2 trunk (grouped
convolutions + candidate selector) and adds a first-token path: the verifier's
native MTP block proposes T1 each round, and a zero-initialized residual
fusion folds the MTP hidden state into T1's slot embedding:

    delta = W_f [RMS_e(e_T1); RMS_m(h_MTP)]
    e~_T1 = e_T1 + delta

The draft block is [anchor, T1, MASK x (B-2)]: the anchor is the bonus token,
slot 1 carries T1 (proposed by the frozen MTP block before the DFlash2 pass),
and the remaining slots are drafted by DFlash2 with the selector walk seeded
by T1 as the first predecessor.

The frozen MTP block is rebuilt from the TARGET model's config (the target
must be the verifier the draft was trained against) and its weights are loaded
from the speculators checkpoint's ``mtp_layers.0.*`` tensors.
"""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.v1.worker.gpu.spec_decode.dmtp2.config import DMTP2SpeculatorConfig

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM, DFlash2Qwen3Model
from .utils import maybe_prefix

# Speculators checkpoints store the frozen MTP block under mtp_layers.0.* with
# training-side module names; vLLM's Qwen3.5 MTP predictor uses the original
# HF names. (Inverse of speculators' MTPConverter remap.)
_MTP_EXACT_REMAP: dict[str, str] = {
    "mtp_layers.0.input_proj.weight": "fc.weight",
    "mtp_layers.0.final_layernorm.weight": "norm.weight",
}
_MTP_PREFIX_REMAP: list[tuple[str, str]] = [
    ("mtp_layers.0.hidden_layernorm.", "pre_fc_norm_hidden."),
    ("mtp_layers.0.token_layernorm.", "pre_fc_norm_embedding."),
    ("mtp_layers.0.", "layers.0."),
]

_SUPPORTED_MTP_MODEL_TYPES = frozenset(
    {"qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text"}
)


class MTPTokenFusion(nn.Module):
    """Zero-initialized residual fusion of T1's embedding with its MTP hidden.

    Mirrors speculators' ``MTPTokenFusion``: each input gets its own RMSNorm
    before a ``2d -> d`` projection whose output is added to the raw token
    embedding. The checkpoint supplies the trained weights; the zero init only
    matters at training time.
    """

    def __init__(
        self,
        hidden_size: int,
        rms_norm_eps: float,
        params_dtype: torch.dtype,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.emb_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.mtp_norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.fusion_proj = ReplicatedLinear(
            2 * hidden_size,
            hidden_size,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "fusion_proj"),
            return_bias=False,
        )

    def forward(
        self,
        token_embedding: torch.Tensor,  # [..., hidden_size]
        mtp_hidden: torch.Tensor,  # [..., hidden_size]
    ) -> torch.Tensor:
        fused = torch.cat(
            [
                self.emb_norm(token_embedding),
                self.mtp_norm(mtp_hidden.to(token_embedding.dtype)),
            ],
            dim=-1,
        )
        return token_embedding + self.fusion_proj(fused)


class DMTP2Qwen3Model(DFlash2Qwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        self.fusion = MTPTokenFusion(
            hidden_size=self.config.hidden_size,
            rms_norm_eps=self.config.rms_norm_eps,
            params_dtype=vllm_config.model_config.dtype,
            prefix=maybe_prefix(prefix, "fusion"),
        )


class DMTP2Qwen3ForCausalLM(DFlash2Qwen3ForCausalLM):
    # The speculator this is a draft model of (see SpeculativeDraftModel).
    speculator_config_class = DMTP2SpeculatorConfig

    model_cls = DMTP2Qwen3Model

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        target_text_config = vllm_config.model_config.hf_text_config
        if target_text_config.model_type not in _SUPPORTED_MTP_MODEL_TYPES:
            raise NotImplementedError(
                "D-MTP2 currently builds its frozen MTP block from Qwen3.5-"
                f"family targets; got model_type={target_text_config.model_type!r}."
            )
        mtp_hidden = getattr(self.config, "dflash_config", {}).get("mtp_hidden_size")
        if mtp_hidden is not None and mtp_hidden != target_text_config.hidden_size:
            raise ValueError(
                f"The draft's frozen MTP block was trained with hidden_size="
                f"{mtp_hidden} but the target has {target_text_config.hidden_size}; "
                "the target must be the verifier the draft was trained against."
            )

        # The frozen MTP block is the verifier's own: build it from the target
        # config. Its attention layer registers as a draft layer and gets its
        # own (causal) KV cache group.
        from vllm.compilation.backends import set_model_tag
        from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MultiTokenPredictor

        with set_model_tag("dmtp2_mtp"):
            self.mtp = Qwen3_5MultiTokenPredictor(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
            )
        self.mtp.requires_grad_(False)
        # compute_t1_logits emits target-space ids directly; the DFlash2
        # backbone requires the full verifier vocabulary anyway.
        assert self.draft_id_to_target_id is None, (
            "D-MTP2 requires draft_vocab_size == the target vocab size."
        )

    def mtp_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """One frozen-MTP pass over the (shifted) target batch.

        Row t carries (h_t, emb(x_{t+1})) and its output predicts x_{t+2};
        per request, the last valid row's output is the hidden that proposes
        T1 (h_MTP). Running the full batch also keeps the MTP KV in sync.
        """
        return self.mtp(input_ids, positions, hidden_states)

    def compute_t1_logits(self, mtp_hidden: torch.Tensor) -> torch.Tensor:
        # The MTP block shares the verifier LM head, which the DFlash loader
        # already shares into self.lm_head.
        return self.logits_processor(self.lm_head, mtp_hidden)

    def fuse_t1_embedding(
        self, t1_embedding: torch.Tensor, mtp_hidden: torch.Tensor
    ) -> torch.Tensor:
        return self.model.fusion(t1_embedding, mtp_hidden)

    def _mtp_attn_layer_names(self) -> list[str]:
        from vllm.model_executor.layers.attention import Attention

        names = [
            module.layer_name
            for module in self.mtp.modules()
            if isinstance(module, Attention)
        ]
        assert names, "The frozen MTP block registered no attention layers."
        return names

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        # The MTP layers come LAST so the context-KV slot lists built from this
        # order still index the DFlash2 layers 0..L-1 (see
        # DFlashSpeculator._precompute_context_kv).
        return super().get_draft_kv_cache_layer_names() + self._mtp_attn_layer_names()

    def get_draft_attn_causal(self) -> list[bool]:
        causal = super().get_draft_attn_causal()
        causal.extend(True for _ in self._mtp_attn_layer_names())
        return causal

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        trunk_weights: list[tuple[str, torch.Tensor]] = []
        mtp_weights: list[tuple[str, torch.Tensor]] = []
        for name, weight in weights:
            if name.startswith("mtp_layers."):
                mtp_weights.append((self._remap_mtp_key(name), weight))
            else:
                # fusion.* flows through the parent loader as model.fusion.*.
                trunk_weights.append((name, weight))
        if mtp_weights:
            self.mtp.load_weights(mtp_weights)
        super().load_weights(trunk_weights)

    @staticmethod
    def _remap_mtp_key(name: str) -> str:
        if name in _MTP_EXACT_REMAP:
            return _MTP_EXACT_REMAP[name]
        for src, dst in _MTP_PREFIX_REMAP:
            if name.startswith(src):
                return dst + name[len(src) :]
        return name


EntryClass = DMTP2Qwen3ForCausalLM
