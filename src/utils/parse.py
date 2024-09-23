def get_attr_through_list(obj, attr_list):
    try:
        for attr in attr_list:
            if isinstance(attr, str):
                obj = getattr(obj, attr)
            elif isinstance(attr, int):
                obj = obj[attr]
            else:
                raise ValueError(f"Invalid type {type(attr)} for {attr}")
            return obj

    except AttributeError as e:
        print(f"{obj} has no attribute {attr}, has attributes: {obj.__dict__.keys()}")
        raise e


def match_parameters(func, args):
    import inspect

    kwargs = {}
    parameters = inspect.signature(func).parameters.keys()
    if "args" in parameters and "kwargs" in parameters:
        return func(**args)

    for key in parameters:
        kwargs[key] = args[key]
    return func(**kwargs)
