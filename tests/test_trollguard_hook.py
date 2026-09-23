"""Tests for TrollGuard OpenClaw hook behavior.

These tests focus on the security-critical `_embed()` fail-closed contract
and the Layer 4 scanner integration path.  They intentionally mock ng_embed
and the OpenClawAdapter base class so the suite runs without downloading
models or starting threads.

# ---- Changelog ----
# [2026-09-22] Claude Code (Kimi k2.7) — Z11 row-1 repair-forward: _embed fail-closed tests.
#   What: Added tests proving _embed() re-raises EmbeddingUnavailableError / DualPassIncompleteError
#         and never falls back to _hash_embed().  Also covers scanner-degradation paths.
#   Why:  Canonical ng_embed is now fail-closed; TrollGuard must never fabricate embeddings.
#   How:  Import-level mocking of ng_embed + OpenClawAdapter.__init__ + threading.Thread.
# -------------------
"""

import logging
import sys
from unittest import mock

import numpy as np
import pytest


@pytest.fixture
def hook_class(monkeypatch):
    """Return the TrollGuardHook class with side effects disabled."""
    # Prevent ng_updater.auto_update from running on import.
    monkeypatch.setattr("ng_updater.auto_update", lambda: None, raising=False)

    import trollguard_hook as tgh

    # Prevent base __init__ from touching filesystem/ecosystem.
    monkeypatch.setattr(tgh.OpenClawAdapter, "__init__", lambda self: None)

    # Prevent any stray thread starts.
    monkeypatch.setattr(tgh.threading.Thread, "start", lambda self: None)

    return tgh.TrollGuardHook


def _make_hook(hook_class, scanner=None):
    """Instantiate hook without side effects."""
    hook = hook_class.__new__(hook_class)
    hook_class.__init__(hook)
    hook._scanner = scanner
    hook._scan_count = 0
    hook._threat_count = 0
    hook._autonomic_last_state = "PARASYMPATHETIC"
    hook._eco = None
    return hook


def test_embed_delegates_to_ng_embed(hook_class, monkeypatch):
    """Normal embed path returns the vector produced by ng_embed.embed."""
    expected = np.ones((768,), dtype=np.float32)

    def fake_embed(text, normalize=False, is_query=False):
        assert text == "hello world"
        return expected

    monkeypatch.setattr("ng_embed.embed", fake_embed)

    hook = _make_hook(hook_class)
    result = hook._embed("hello world")

    assert result is expected
    assert result.shape == (768,)


def test_embed_fails_closed_on_unavailable(hook_class, monkeypatch, caplog):
    """EmbeddingUnavailableError is logged and re-raised, never swallowed."""
    from ng_embed import EmbeddingUnavailableError

    def fake_embed(text, normalize=False, is_query=False):
        raise EmbeddingUnavailableError("model offline")

    monkeypatch.setattr("ng_embed.embed", fake_embed)

    hook = _make_hook(hook_class)
    caplog.set_level(logging.ERROR, logger="trollguard_hook")

    with pytest.raises(EmbeddingUnavailableError, match="model offline"):
        hook._embed("hello world")

    assert any(
        "EmbeddingUnavailableError" in rec.message and "no hash fallback" in rec.message
        for rec in caplog.records
    )


def test_embed_fails_closed_on_dualpass_incomplete(hook_class, monkeypatch, caplog):
    """DualPassIncompleteError is also propagated, not swallowed as generic Exception."""
    from ng_embed import DualPassIncompleteError

    def fake_embed(text, normalize=False, is_query=False):
        raise DualPassIncompleteError("pass-2 TID absent")

    monkeypatch.setattr("ng_embed.embed", fake_embed)

    hook = _make_hook(hook_class)
    caplog.set_level(logging.ERROR, logger="trollguard_hook")

    with pytest.raises(DualPassIncompleteError, match="pass-2 TID absent"):
        hook._embed("hello world")

    assert any(
        "DualPassIncompleteError" in rec.message and "no hash fallback" in rec.message
        for rec in caplog.records
    )


def test_embed_does_not_catch_unexpected_errors(hook_class, monkeypatch):
    """Non-canonical exceptions are not caught by _embed(); bubble to caller."""
    from ng_embed import DualPassIncompleteError, EmbeddingUnavailableError

    def fake_embed(text, normalize=False, is_query=False):
        raise ValueError("unexpected internal error")

    monkeypatch.setattr("ng_embed.embed", fake_embed)

    hook = _make_hook(hook_class)

    with pytest.raises(ValueError, match="unexpected internal error"):
        hook._embed("hello world")


def test_embed_never_falls_back_to_hash_embed(hook_class, monkeypatch):
    """_hash_embed must never be invoked regardless of embed failure type."""
    from ng_embed import EmbeddingUnavailableError

    def fake_embed(text, normalize=False, is_query=False):
        raise EmbeddingUnavailableError("model offline")

    monkeypatch.setattr("ng_embed.embed", fake_embed)

    hook = _make_hook(hook_class)

    with mock.patch.object(hook_class, "_hash_embed") as mock_hash:
        with pytest.raises(EmbeddingUnavailableError):
            hook._embed("hello world")
        mock_hash.assert_not_called()


def test_module_on_message_degraded_when_scanner_unavailable(hook_class):
    """_module_on_message reports scanner_unavailability when pipeline is down."""
    hook = _make_hook(hook_class, scanner=None)
    dummy_emb = np.ones((768,), dtype=np.float32)

    result = hook._module_on_message("safe text", dummy_emb)

    assert result["scan_status"] == "scanner_unavailable"
    assert result["scan_count"] == 1
