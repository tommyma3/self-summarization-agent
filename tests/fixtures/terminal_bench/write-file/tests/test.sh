#!/bin/sh
mkdir -p /logs/verifier
python3 - <<'PY'
from pathlib import Path
path = Path('/workspace/result.txt')
passed = path.is_file() and path.read_bytes() == b'hello\n'
Path('/logs/verifier/reward.txt').write_text('1' if passed else '0')
PY
