"""R2 CLI entry point."""
from __future__ import annotations

import sys
from gated_dual_ema_msd.training.r2_protocol import build_parser, main as protocol_main


def main(argv: list[str] | None = None) -> dict:
    if argv is None:
        argv = sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(argv)
    return protocol_main(args)


if __name__ == "__main__":
    main()
