from pathlib import Path

from tools.agentic import gpt_bundle, project_state_refresh


def _is_gpt_bundle_scratch(repo: Path, path: Path) -> bool:
    try:
        rel = path.relative_to(repo)
    except ValueError:
        return False
    if len(rel.parts) < 3:
        return False
    return tuple(rel.parts[:3]) == ("artifacts", "_local", "gpt_bundles")


def test_agentic_bundle_defaults_are_in_scratch(tmp_path: Path) -> None:
    repo = tmp_path
    ts = "20260126_000000"
    assert _is_gpt_bundle_scratch(repo, gpt_bundle.default_bundle_path(repo, ts, "TICKET-123"))
    assert _is_gpt_bundle_scratch(repo, project_state_refresh.default_zip_path(repo, ts))


def test_stash_wrapper_roundtrip(monkeypatch, tmp_path: Path) -> None:
    repo = tmp_path
    status_before = " M foo.py\n?? bar.txt\n"
    status_sequence = iter([status_before, "", status_before])

    def fake_run(cmd: list[str], cwd: Path | None = None) -> tuple[int, str]:
        if cmd[:4] == ["git", "-C", str(repo), "status"]:
            return 0, next(status_sequence)
        if cmd[:5] == ["git", "-C", str(repo), "stash", "push"]:
            return 0, "Saved working directory"
        if cmd[:5] == ["git", "-C", str(repo), "stash", "list"]:
            return 0, "stash@{0}: On main: temp: gpt_bundle TICKET-TEST\n"
        if cmd[:5] == ["git", "-C", str(repo), "stash", "apply"]:
            return 0, ""
        if cmd[:5] == ["git", "-C", str(repo), "stash", "drop"]:
            return 0, "Dropped stash@{0}\n"
        return 0, ""

    monkeypatch.setattr(gpt_bundle, "run", fake_run)

    dirty, stash_used, seen_status, stash_ref, err = gpt_bundle.stash_if_dirty(
        repo, "TICKET-TEST", "20260126_000000", allow_stash=True
    )
    assert dirty is True
    assert stash_used is True
    assert seen_status == status_before
    assert stash_ref == "stash@{0}"
    assert err is None

    restore_err = gpt_bundle.restore_stash(repo, stash_ref, status_before)
    assert restore_err is None
