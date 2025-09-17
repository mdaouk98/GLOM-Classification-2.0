# src/main_utils/helpers_utils/timeit.py

import time
from contextlib import contextmanager

@contextmanager
def timeit():
    """Context manager to measure the elapsed time of a code block."""
    start = time.time()
    yield lambda: time.time() - start


