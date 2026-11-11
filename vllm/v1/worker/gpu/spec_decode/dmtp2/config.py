# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.v1.worker.gpu.spec_decode.dflash2.config import DFlash2SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


class DMTP2SpeculatorConfig(DFlash2SpeculatorConfig):
    """D-MTP2: DFlash2 backbone whose draft block starts from a frozen-MTP-
    proposed first token (dual-anchor layout [A, T1, MASK x (B-2)])."""

    base_method = "dmtp2"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.dmtp2.speculator.DMTP2Speculator"
    )
    # The frozen-MTP first-token pass must run every step to keep the MTP KV
    # cache in sync, so context-KV-only prefill is not supported.
    supports_prefill_context_kv_only = False
    supports_watermarking = False

    @classmethod
    def matches(cls, spec: "SpeculativeConfig") -> bool:
        return "dmtp" in spec.draft_model_config.model.lower()
