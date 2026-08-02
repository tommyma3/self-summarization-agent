from __future__ import annotations

from pathlib import Path

from self_summarization_agent.checkpoints import (
    move_incomplete_checkpoint_aside,
    publish_checkpoint,
)
from self_summarization_agent.train_step import run_train_step


def run_train_update(
    config,
    *,
    checkpoint_path: str | Path,
    rollout_path: str | Path,
    output_checkpoint_path: str | Path,
    metrics_path: str | Path | None = None,
) -> Path:
    """Run one update into a temporary directory, then publish it atomically."""
    output_checkpoint = Path(output_checkpoint_path)
    partial_checkpoint = output_checkpoint.with_name(f".{output_checkpoint.name}.incomplete")
    move_incomplete_checkpoint_aside(partial_checkpoint)
    trained = run_train_step(
        config,
        checkpoint_path=checkpoint_path,
        rollout_path=rollout_path,
        output_checkpoint_path=partial_checkpoint,
        metrics_path=metrics_path,
        published_checkpoint_path=output_checkpoint,
    )
    return publish_checkpoint(Path(trained), output_checkpoint)
