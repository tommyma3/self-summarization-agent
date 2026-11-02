def is_harbor(config):
    return getattr(getattr(config, "benchmark", config), "name", "browsecomp") in {"terminal-bench", "swe-dev"}


def task_manifest(config, *, split):
    if config.name == "swe-dev":
        from ..swe_dev.dataset import task_manifest as resolve
    else:
        from ..terminal_bench.dataset import task_manifest as resolve
    return resolve(config, split=split)


def benchmark_identity(config):
    if config.name == "swe-dev":
        from ..swe_dev.dataset import benchmark_identity as identity
    else:
        from ..terminal_bench.dataset import benchmark_identity as identity
    return identity(config)


def selected_tasks(config, *, split, sample_seed=None):
    import random
    tasks = task_manifest(config.benchmark, split=split)
    # Eval membership/order is fixed independently of iteration/sample seeds.
    seed = config.experiment.seed if sample_seed is None or split == "eval" else sample_seed
    rng = random.Random(seed)
    if config.dataset.shuffle and split == "train":
        rng.shuffle(tasks)
    tasks = tasks[config.dataset.offset:]
    if config.dataset.limit is not None:
        tasks = tasks[:config.dataset.limit]
    count = getattr(config.collection, f"{split}_task_count")
    if count is not None:
        if count < 0 or count > len(tasks) or (split == "train" and count == 0):
            raise ValueError("Requested task count is outside the selected benchmark")
        tasks = tasks[:count] if split == "eval" else rng.sample(tasks, count)
    return tasks
