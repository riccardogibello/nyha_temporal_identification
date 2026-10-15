"""Tests for scripts/max_cwin_calculator.py — KV geometry, dense/SWA concurrency,
vLLM layer-group padding, and the observed-point calibration. Numbers are
anchored to a real vLLM v0.11 startup for gemma-3-27b-w4a16 on an H100:
    Available KV cache memory: 63.38 GiB
    Add 8 padding layers, may waste at most 15.38% KV cache memory
    GPU KV cache size: 118,672 tokens
    Maximum concurrency for 25,200 tokens per request: 10.31x
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "max_cwin_calculator.py"
_spec = importlib.util.spec_from_file_location("max_cwin_calculator", _SCRIPT)
mcw = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcw)

# gemma-3-27b geometry
N_LAYERS = 62
N_GLOBAL = N_LAYERS // 6  # sliding_window_pattern = 6 -> 10 global
N_KV_HEADS = 16
HEAD_DIM = 128
WINDOW = 1024
CAPACITY = 118_672  # vLLM "GPU KV cache size"
CTX = 25_200
OBSERVED_CONC = 10.31


def test_per_token_layer_bytes():
    # 2 (K+V) * 16 heads * 128 dim * 2 bytes (bf16)
    assert mcw.per_token_layer_bytes(N_KV_HEADS, HEAD_DIM, 2) == 8192


def test_layer_group_padding_matches_vllm():
    # 52 sliding layers padded up to a multiple of 10 globals -> 60, total 70.
    n_local_padded, n_layers_padded = mcw.padded_layer_counts(N_LAYERS, N_GLOBAL)
    assert n_local_padded == 60
    assert n_layers_padded == 70
    # Waste warning: 8 padding layers over 52 real local layers = 15.38%.
    waste = (n_local_padded - (N_LAYERS - N_GLOBAL)) / (N_LAYERS - N_GLOBAL)
    assert round(100 * waste, 2) == 15.38


def test_no_padding_for_dense_model():
    assert mcw.padded_layer_counts(32, 0) == (0, 32)


def test_dense_concurrency_is_safe_floor():
    # Dense: footprint == ctx, so concurrency == capacity / ctx == vLLM's naive 4.71x.
    conc = mcw.concurrency_at_ctx(
        CAPACITY, CTX, n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70, window=None
    )
    assert round(conc, 2) == 4.71


def test_swa_ideal_is_optimistic_ceiling():
    # Ideal sliding-window credit overshoots vLLM's realized 10.31x (~26.5x).
    conc = mcw.concurrency_at_ctx(
        CAPACITY, CTX, n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70, window=WINDOW
    )
    assert conc > 25
    assert conc > OBSERVED_CONC  # the trap: ideal SWA is far above reality


def test_calibration_reproduces_observed_concurrency():
    w_eff = mcw.calibrate_effective_window(
        CAPACITY, OBSERVED_CONC, CTX,
        n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70,
    )
    # Realized window sits well above the configured 1024 (partial SWA credit).
    assert WINDOW < w_eff < CTX
    # Round-trips: feeding w_eff back yields the observed concurrency.
    conc = mcw.concurrency_at_ctx(
        CAPACITY, CTX, n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70, window=w_eff
    )
    assert round(conc, 2) == OBSERVED_CONC


def test_calibrated_concurrency_brackets_between_dense_and_swa():
    w_eff = mcw.calibrate_effective_window(
        CAPACITY, OBSERVED_CONC, CTX,
        n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70,
    )
    geom = dict(n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70)
    dense = mcw.concurrency_at_ctx(CAPACITY, CTX, window=None, **geom)
    swa = mcw.concurrency_at_ctx(CAPACITY, CTX, window=WINDOW, **geom)
    cal = mcw.concurrency_at_ctx(CAPACITY, CTX, window=w_eff, **geom)
    assert dense < cal < swa


def test_max_ctx_inverse_roundtrip():
    geom = dict(n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70, window=WINDOW)
    ctx = mcw.max_ctx_at_concurrency(CAPACITY, 10.0, **geom)
    # Pushing exactly that ctx back through gives ~10x.
    conc = mcw.concurrency_at_ctx(CAPACITY, ctx, **geom)
    assert round(conc, 1) == 10.0


def test_dense_inverse_is_simple_division():
    # window=None -> max ctx is just capacity / concurrency.
    ctx = mcw.max_ctx_at_concurrency(
        CAPACITY, 10.0, n_global=N_GLOBAL, n_local_padded=60, n_layers_padded=70, window=None
    )
    assert ctx == int(CAPACITY / 10.0)
