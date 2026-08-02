from pathlib import Path

from self_summarization_agent.checkpoints import (
    advance_latest_checkpoint,
    is_vllm_loadable_checkpoint,
    mark_checkpoint_complete,
    publish_checkpoint,
    resolve_latest_checkpoint,
)


def write_fake_checkpoint(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_text("weights", encoding="utf-8")
    mark_checkpoint_complete(path)


def test_latest_checkpoint_advances_only_for_complete_loadable_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "step-00001"
    write_fake_checkpoint(checkpoint)

    advanced = advance_latest_checkpoint(tmp_path, checkpoint)

    assert advanced.checkpoint_id == "step-00001"
    assert resolve_latest_checkpoint(tmp_path).path == checkpoint.resolve()
    assert is_vllm_loadable_checkpoint(checkpoint)


def test_latest_checkpoint_rejects_partial_checkpoint(tmp_path: Path) -> None:
    partial = tmp_path / "checkpoints" / "partial"
    partial.mkdir(parents=True)
    (partial / "config.json").write_text("{}", encoding="utf-8")

    try:
        advance_latest_checkpoint(tmp_path, partial)
    except ValueError as exc:
        assert "not complete or vLLM-loadable" in str(exc)
    else:
        raise AssertionError("Expected partial checkpoint to be rejected")


def test_publish_checkpoint_moves_aside_an_incomplete_destination(tmp_path: Path) -> None:
    partial = tmp_path / ".iteration-00001.incomplete"
    destination = tmp_path / "iteration-00001"
    write_fake_checkpoint(partial)
    destination.mkdir()
    (destination / "partial.txt").write_text("partial", encoding="utf-8")

    published = publish_checkpoint(partial, destination)

    assert published == destination
    assert is_vllm_loadable_checkpoint(destination)
    assert (tmp_path / "iteration-00001.stale-001" / "partial.txt").exists()
