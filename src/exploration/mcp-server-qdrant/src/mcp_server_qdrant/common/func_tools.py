# danish: inspect is a built-in library in python
import inspect
from functools import wraps
from typing import Callable


# danish: this function pre-fills specific arguments similar to functool.partial, but by modifying the signature of the function it then also ensures that tools like FastAPI which might read the signature of a function, don't see those fixed / filled in values in the signature
def make_partial_function(original_func: Callable, fixed_values: dict) -> Callable:
    # danish: this gives you access to the parameters of a function
    sig = inspect.signature(original_func)

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        # Start with fixed values
        bound_args = dict(fixed_values)

        # Bind positional/keyword args from caller
        for name, value in zip(remaining_params, args):
            bound_args[name] = value
        bound_args.update(kwargs)

        return original_func(**bound_args)

    # Only keep parameters NOT in fixed_values
    remaining_params = [name for name in sig.parameters if name not in fixed_values]
    new_params = [sig.parameters[name] for name in remaining_params]

    # Set the new __signature__ for introspection
    wrapper.__signature__ = sig.replace(parameters=new_params)  # type:ignore

    return wrapper
