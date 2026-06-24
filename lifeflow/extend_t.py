def extend_t(T):
    def decorator(func):
        def wrapper():
            return [func(t) for t in range(T)]

        return wrapper

    return decorator
