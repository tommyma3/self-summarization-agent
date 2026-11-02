"""Trusted fixed discovery. Candidate code runs only in child processes.

The parent owns expected values and counts. Candidate stdout is a function result,
never a reward/discovery report. No candidate test/config module is discovered.
"""
import json
from pathlib import Path
import subprocess
import sys

CHILD = """
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location('candidate_names', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps(module.normalize_name(json.loads(sys.argv[2]))))
"""
cases = [("  Ada  ", "Ada"), ("a  b", "a  b"), ("", "")]
results = []
for source, expected in cases:
    try:
        result = subprocess.run([sys.executable, '-I', '-S', '-c', CHILD,
            str(Path(sys.argv[1]) / 'names.py'), json.dumps(source)],
            capture_output=True, text=True, timeout=2)
        results.append(result.returncode == 0 and json.loads(result.stdout) == expected)
    except (ValueError, subprocess.TimeoutExpired):
        results.append(False)
print(json.dumps({"expected": 3, "executed": len(results), "passed": sum(results)}))
sys.exit(0 if all(results) else 1)
