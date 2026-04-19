import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from self_summarization_agent.cli import main as cli_main


def main() -> None:
    cli_main()


if __name__ == "__main__":
    main()
