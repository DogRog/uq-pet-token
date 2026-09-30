"""Grid-search fully supervised sentence training on validation, then evaluate it on test."""

import argparse

from uq_pet import supervised


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    supervised.configure_parser(parser)
    supervised.run(parser.parse_args(argv), parser)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
