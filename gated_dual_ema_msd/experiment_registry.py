"""Forwarding alias for experiment_registry."""
import importlib
import sys
sys.modules[__name__] = importlib.import_module("gated_dual_ema_msd.config.experiments")
