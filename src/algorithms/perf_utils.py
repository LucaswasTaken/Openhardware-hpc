"""
Performance utility functions extracted from the notebook.

This module provides small helpers used throughout the Aula2 notebook:
- time_function: simple timing decorator
- profile_function: run cProfile and print top results
- compute_intensive_task: small CPU-bound function for demonstrations

Keep these functions lightweight and well-documented so the notebook stays
readable and imports are fast.
"""
from functools import wraps
import time
import cProfile
import pstats
import io
import numpy as np

__all__ = [
    "time_function",
    "profile_function",
    "compute_intensive_task",
]


def time_function(func):
    """Decorator to time a function using time.perf_counter().

    Prints the function name and elapsed time when the wrapped function
    completes. Intended for quick demonstrations in notebooks.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.4f}s")
        return result

    return wrapper


def profile_function(func, *args, **kwargs):
    """Run a function under cProfile and print the top 10 cumulative calls.

    Returns the function result. The profiling output is printed to stdout.
    Useful for quick performance investigations in teaching notebooks.
    """
    pr = cProfile.Profile()
    pr.enable()
    result = func(*args, **kwargs)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(10)

    print("📊 Profile detalhado:")
    print(s.getvalue())
    return result


@time_function
def compute_intensive_task(n):
    """Example CPU-bound work: sum of squares up to n.

    Kept intentionally simple so it's deterministic and fast to run in
    demonstration contexts. Decorated with `time_function` so calling it
    prints the elapsed time automatically.
    """
    total = 0
    # use local variable for speed
    for i in range(n):
        total += i * i
    return total
