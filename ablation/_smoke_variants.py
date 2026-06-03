"""Smoke-test every DFlash architecture variant: build the model + run a
forward/backward on random tensors, asserting finite loss and correct shapes.

Run with:  TORCH_COMPILE_DISABLE=1 .venv/bin/python ablation/_smoke_variants.py
(TORCH_COMPILE_DISABLE keeps it fast; the forward is @torch.compile in prod.)
"""

import sys

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

import speculators.models  # noqa: F401  (registers simple_flex_attention)
from speculators.config import SpeculatorsConfig, VerifierConfig
from speculators.models.dflash import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.proposals.greedy import GreedyTokenProposalConfig

HIDDEN = 256
N_AUX = 5  # baseline aux layers 1 9 17 25 34
VOCAB = 1000
SEQ = 64
DEVICE = "cuda"


def tiny_tl_config(
    num_layers=2,
    intermediate_size=512,
    num_heads=4,
    num_kv_heads=2,
    decoder_use_mlp=True,
):
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_layers,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=HIDDEN // num_heads,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    cfg.decoder_use_mlp = decoder_use_mlp
    return cfg


def build(tl_config, fusion_type="linear", draft_sliding_window=None):
    config = DFlashSpeculatorConfig(
        transformer_layer_config=tl_config,
        draft_vocab_size=VOCAB,
        block_size=4,
        max_anchors=8,
        aux_hidden_state_layer_ids=[1, 9, 17, 25, 34],
        mask_token_id=0,
        fusion_type=fusion_type,
        draft_sliding_window=draft_sliding_window,
        speculators_config=SpeculatorsConfig(
            algorithm="dflash",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=3)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="tiny", architectures=["Qwen3"]),
        ),
    )
    model = DFlashDraftModel(config=config)
    # The model nan-inits embed/lm_head/verifier_lm_head to flag unloaded weights;
    # fill all non-finite params so a forward is testable without real checkpoints.
    with torch.no_grad():
        for p in model.parameters():
            if not torch.isfinite(p).all():
                p.normal_(0.0, 0.02)
    # from_training_args sets these training-only attrs; PreTrainedModel otherwise
    # provides loss_type=None, so set them here to mirror the real training path.
    model.loss_type = "ce"
    model.loss_gamma = 4.0
    model.label_smoothing = 0.0
    return model.to(DEVICE).train()


def run_forward(model):
    hs = torch.randn(1, SEQ, N_AUX * HIDDEN, device=DEVICE)
    input_ids = torch.randint(0, VOCAB, (1, SEQ), device=DEVICE)
    loss_mask = torch.ones(1, SEQ, dtype=torch.long, device=DEVICE)
    vlast = torch.randn(1, SEQ, HIDDEN, device=DEVICE)
    lengths = torch.tensor([SEQ], dtype=torch.long, device=DEVICE)
    draft_tokens, loss, metrics = model(
        hidden_states=hs,
        input_ids=input_ids,
        loss_mask=loss_mask,
        verifier_last_hidden_states=vlast,
        lengths=lengths,
    )
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()  # exercise the backward graph
    return float(loss), tuple(draft_tokens.shape)


CASES = [
    ("baseline", lambda: build(tiny_tl_config())),
    ("depth-1", lambda: build(tiny_tl_config(num_layers=1))),
    ("width-half", lambda: build(tiny_tl_config(intermediate_size=256))),
    ("fusion-mlp", lambda: build(tiny_tl_config(), fusion_type="mlp")),
    ("fusion-gated", lambda: build(tiny_tl_config(), fusion_type="gated")),
    ("fusion-wsum", lambda: build(tiny_tl_config(), fusion_type="weighted_sum")),
    ("no-decoder-mlp", lambda: build(tiny_tl_config(decoder_use_mlp=False))),
    ("heads-2-1", lambda: build(tiny_tl_config(num_heads=2, num_kv_heads=1))),
    ("swa-16", lambda: build(tiny_tl_config(), draft_sliding_window=16)),
]


def main() -> int:
    failures = []
    for name, factory in CASES:
        try:
            model = factory()
            # sanity: no-decoder-mlp layers must lack an mlp attr
            if name == "no-decoder-mlp":
                assert not hasattr(model.layers[0], "mlp"), "mlp present despite toggle"
            if name == "fusion-mlp":
                assert type(model.fc).__name__ == "_MLPFusion"
            loss, shape = run_forward(model)
            print(f"OK   {name:16s} loss={loss:.4f} draft_tokens={shape}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name:16s} {type(e).__name__}: {e}")
            failures.append(name)
    if failures:
        print(f"\n{len(failures)} FAILED: {failures}")
        return 1
    print("\nAll variant smoke tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
