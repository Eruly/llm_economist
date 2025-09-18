"""
LLM Economist: A framework for economic simulations using Large Language Models.
"""

__version__ = "1.0.0"
__author__ = "Seth Karten, Wenzhe Li, Zihan Ding, Samuel Kleiner, Yu Bai, Chi Jin"

__all__ = ["run_simulation"]


def run_simulation(*args, **kwargs):
    from .main import run_simulation as _run_simulation

    return _run_simulation(*args, **kwargs)
