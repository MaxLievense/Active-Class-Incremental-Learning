def expection_wrapper(func):
    def wrapper(*args, **kwargs):
        try:
            output = func(*args, **kwargs)
        except Exception as e:
            return e

    return wrapper
