"""Forwarding alias for trainer."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("gated_dual_ema_msd.training.direct")
