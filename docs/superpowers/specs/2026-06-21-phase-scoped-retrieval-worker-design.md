# Phase-Scoped Retrieval Worker Design

## Goal

Free the FAISS embedding model's CUDA memory before judging, rollout caching, and the policy weight update. The embedding model should be resident only while train or evaluation rollouts can issue retrieval requests.

## Current Behavior

With `retrieval.persistent_worker: true`, `iteration_launcher` starts one retrieval worker before the train rollout and stops it only after the complete iteration. The worker is reused by the evaluation rollout, but Qwen3-Embedding-8B remains resident during train judging, caching, the policy update, evaluation judging, and metric calculation.

## Selected Design

Keep the retrieval worker architecture and narrow each worker's lifetime to one rollout phase:

1. If the train rollout is incomplete, start a retrieval worker, pass its URL to the train rollout command, run the rollout, and stop the worker in a `finally` block.
2. Run train judging, caching, and the policy update with no retrieval worker alive.
3. If evaluation is enabled and the eval rollout is incomplete, start a new retrieval worker, pass its URL to the eval rollout command, run the rollout, and stop the worker in a `finally` block.
4. Run evaluation judging and metric calculation with no retrieval worker alive.

When `retrieval.persistent_worker` is false, rollout collection retains its current direct-backend behavior. The change does not alter retrieval configuration, query results, rollout artifacts, or phase ordering.

## Resume and Failure Behavior

Resume checks happen before worker startup. A completed train or eval rollout must not load the embedding model. Each worker is stopped even if startup succeeds but its rollout subprocess fails. A train-rollout failure prevents later phases as it does today; an eval-rollout failure prevents eval judging and metrics as it does today.

## Trade-off

The embedding model and FAISS backend are loaded twice in an iteration that performs both train and eval rollouts. This adds startup latency compared with sharing one worker, but it guarantees that the embedding model does not consume CUDA memory during the policy update.

## Validation

Launcher tests will record phase events and verify:

- one worker lifecycle surrounds an incomplete train rollout;
- a separate worker lifecycle surrounds an incomplete eval rollout;
- the train update occurs after the train worker stops and before the eval worker starts;
- completed rollout phases do not start workers during resume;
- rollout commands receive only the URL of their phase-local worker;
- workers are stopped when rollout execution fails.

Focused launcher tests and Python compile checks will validate the implementation.
