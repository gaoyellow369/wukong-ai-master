import time
from log import log
from functools import wraps

def timeit(func):
    """
    Decorator that logs the time a function takes to execute.
    Example：
    
    @timeit
    def my_function():
        pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time() 
        elapsed_time = end_time - start_time
        log(f"{func.__name__} elapsed: {elapsed_time:.6f} s")
    return wrapper
