"""Compatibility entry point."""
import importlib
import sys
implementation = importlib.import_module("gated_dual_ema_msd.cli.train")
if __name__ == "__main__":
    implementation.main()
else:
    sys.modules[__name__] = implementation
