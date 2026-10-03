Terminal-Bench 2.0 with Terminus-KIRA and TITO

The `terminus-kira-tito` scaffold uses Harbor 0.1.44 for isolated task environments and verification. KIRA's prompt, tool descriptions, terminal polling, and completion checks are adapted from revision `652dacbf14d29ea93a83c496ee91e0e5ba286721`; attribution is packaged with the implementation. The bundled official manifest contains 89 tasks from Terminal-Bench 2.0 revision `69671fbaac6d67a7ef0dfec016cc38a64ef7a77c`.

Policy inference, compaction, exact token evidence, and training masks are owned by the shared project runtime. The upstream chat-history loop and its history-unwinding recovery are not used. BrowseComp remains the default benchmark.

1. **Install with `uv`.** From the repository root:

   ```bash
   uv venv --python 3.12  # Only if .venv does not already exist.
   uv sync --locked --extra terminal-bench
   ```

   Docker must be installed and usable for the supplied environment configs. It is a host prerequisite, not a Python dependency. Offline vLLM requires the usual compatible GPU environment. All commands below use the locked project environment.

2. **Validate the local fixture before running the benchmark.** The fixture writes a file and verifies its exact bytes. On a Docker host:

   ```bash
   uv run --locked --extra terminal-bench harbor run \
     --path tests/fixtures/terminal_bench/write-file --agent oracle --env docker

   uv run --locked --extra terminal-bench python -m self_summarization_agent.rollout_collection \
     --config configs/eval/terminal_bench_smoke.yaml \
     --checkpoint /path/to/Qwen3.5-checkpoint \
     --split eval --output outputs/terminal-smoke/rollouts.jsonl
   ```

   The smoke configuration requests compaction after every completed tool round. A successful summary is required to replace context. A malformed or over-limit summary retains its raw output and terminates with the configured penalty.

3. **Run official evaluation.** Start with a task filter; omit it for all 89 tasks:

   ```bash
   uv run --locked --extra terminal-bench python -m self_summarization_agent.rollout_collection \
     --config configs/eval/terminal_bench.yaml \
     --checkpoint /path/to/Qwen3.5-checkpoint \
     --split eval --output outputs/terminal-eval/rollouts.jsonl \
     --set 'benchmark.task_names=["adaptive-rejection-sampler"]'
   ```

   Use `configs/eval/terminal_bench_no_compaction.yaml` for the matching comparison, with a fresh output directory. Set `evaluation.samples_per_task` for repeated attempts and `rollout.max_concurrent_episodes` for concurrent environments. Shared model access is serialized; terminal operations overlap across trials. Set `rollout.gpu_ids` and the checkpoint for your host.

   The token-capable vLLM Completions endpoint is also supported through `rollout.backend=openai_compatible`, `rollout.api_base_url`, and `rollout.api_model`. It must accept integer prompts and return authoritative token IDs, including sampled termination tokens. Message-only chat endpoints are rejected. Summary and forced-completion constraints are supported by the vLLM paths. Control-turn thinking is constrained to prose without `<` markup, followed by the required closing tag and summary or completion action; raw generated tokens remain untouched. Other existing token backends retain strict runtime validation, but the tested real engine for this adapter is offline vLLM.

4. **Choose the image profile explicitly.** The supplied configs are labeled `text-only` and omit `image_read` from their stable tool schema. To enable the complete tool set, configure:

   ```yaml
   benchmark:
     name: terminal-bench
     image_profile: local-vision
     vision_model_path: /path/to/separate-frozen-vision-checkpoint
     vision_device: cuda:1
     vision_max_tokens: 2048
   ```

   The helper loads only an existing local Transformers image-text checkpoint, with local-files-only loading, evaluation mode, and frozen weights. It must be separate from the changing trainable policy. Images are read from the task environment and analyzed locally; no external model endpoint receives them. Helper text is a linked conditioning-only observation. Image hashes, helper identity, text, and usage are saved separately. Helper generations are not policy training tokens. Direct optimization of multimodal helper reasoning is outside the current text-token trajectory contract.

5. **Inspect and resume artifacts.** Collection writes completed rollout rows, verifier evidence, and a `.metrics.json` companion. Trial directories beside the rollout file contain Harbor results, raw observations, optional vision records, `agent/rollout.json`, and `agent/trajectory.records.jsonl`. Each interval includes exact prompt/completion/full IDs, append spans, assistant masks, and renderer identity. Harbor logs are diagnostic exports, never reconstruction inputs.

   ```bash
   uv run --locked --extra terminal-bench python scripts/verify_tito_records.py \
     /path/to/trial/agent/trajectory.records.jsonl --tokenizer /path/to/checkpoint
   ```

   Add `--resume` to the same collection command to skip recorded attempts. Dataset, scaffold, control text, tools, runtime settings, tokenizer, checkpoint, and image-profile identity are checked before reuse. Interrupted attempts without a committed row start in a fresh environment; they do not resume a shell from reconstructed history. Recorded infrastructure failures remain distinct outcomes and require a fresh run if you want to retry them.

   Raw verifier reward is retained separately. Binary success maps to training reward +1 and failure to -1, while malformed output retains its penalty. Infrastructure or verifier errors are excluded from trainable samples. Every interval from a valid rollout shares its terminal reward. Benchmark pass rate is reported over valid attempts with infrastructure-error counts alongside it; inspect task coverage before comparing scores.

6. **Use the existing training pipeline.** Specify `benchmark.train_task_paths` with explicit local Harbor task directories. Official benchmark tasks are evaluation-only by default; `benchmark.allow_benchmark_training=true` is an explicit opt-in and those results must be labeled accordingly. The merged collection path runs isolated policy processes, imports stored verifier rewards without a model judge, and builds existing exact-token caches. It starts no BrowseComp retrieval or judge workers. GRPO normalization and value-MC interval anchors remain unchanged. Use the usual training configs with the new benchmark block and compatible rollout/context ceilings.

   A standalone Harbor entry point is available as `self_summarization_agent.benchmarks.terminal_bench.harbor_agent:TerminusKiraTito`, accepting `config_path` as an agent kwarg. The project collection command above additionally provides grouped attempts, verifier-to-training conversion, and resume lineage checks.

7. **Run transport checks without Docker.** This probe supplies synthetic observations and executes no model-generated commands. It is not a benchmark score:

   ```bash
   CUDA_VISIBLE_DEVICES=0 uv run --locked --extra terminal-bench python \
     scripts/probe_terminal_bench_tito.py --model /path/to/Qwen3.5-checkpoint \
     --output-dir /tmp/terminal-tito-probe

   TITO_TEST_TOKENIZER_PATH=/path/to/Qwen3.5-checkpoint \
     uv run --locked --extra terminal-bench pytest tests/test_terminal_bench.py tests/test_tito.py -q
   ```

   On 2026-10-03, the local Qwen3.5-9B/vLLM probe completed one interval without compaction and five intervals with four successful summaries and forced completion. The full suite passed 313 tests, including real-tokenizer tests, verifier/cache integration, malformed output, resume rejection, concurrency, and CPU updates for both RL objectives. Docker/oracle/official task execution and real local-vision inference remain unverified on this host because Docker is unavailable and no separate frozen vision checkpoint was selected for a run.
