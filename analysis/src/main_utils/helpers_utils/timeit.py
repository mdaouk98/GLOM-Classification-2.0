# src/main_utils/helpers_utils/timeit.py

import time
from contextlib import contextmanager

@contextmanager
def timeit():
    """
    Context manager for timing a block of code.

    Usage:
        with timeit() as elapsed:
            # ... code to measure ...
        print(f"Elapsed time: {elapsed():.4f} seconds")

    Yields:
        A zero-argument callable that returns elapsed seconds since entering the block.
    """
    # 1) Record start time
    start = time.time()
    try:
        # 2) Yield a function that computes elapsed time when called
        yield lambda: time.time() - start
    finally:
        # (Optional) Could add cleanup or logging here if desired
        pass



