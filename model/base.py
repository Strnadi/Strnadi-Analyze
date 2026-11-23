"""Base class of every model in `model/`.

It does one thing: remember the arguments a model was constructed with, so that a checkpoint can rebuild the
exact same architecture later without the experiment script that created it (`training/checkpoint.py`).
"""

import functools
import inspect

import torch.nn as nn


class Model(nn.Module):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        original_init = cls.__init__

        @functools.wraps(original_init)
        def init_and_remember(self, *args, **kw):
            # Only the outermost constructor call records its arguments (a subclass calls its parent's __init__).
            if '_init_args' not in self.__dict__:
                bound = inspect.signature(original_init).bind(self, *args, **kw)
                arguments = dict(list(bound.arguments.items())[1:])
                for name, parameter in inspect.signature(original_init).parameters.items():
                    if parameter.kind is inspect.Parameter.VAR_KEYWORD and name in arguments:
                        arguments.update(arguments.pop(name))
                object.__setattr__(self, '_init_args', arguments)
            original_init(self, *args, **kw)

        cls.__init__ = init_and_remember

    @property
    def init_args(self):
        """Constructor arguments as given (defaults not included)."""
        return dict(self.__dict__.get('_init_args', {}))

    @property
    def class_path(self):
        return f'{type(self).__module__}.{type(self).__qualname__}'


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def describe(model, example=None):
    """Print the size of a model and the shapes it produces for 4 s of audio (or for `example`)."""
    import torch

    if example is None:
        example = torch.randn(2, 4 * 32000) * 0.1
    model.eval()
    with torch.no_grad():
        outputs = model(example)
    print(f'{type(model).__name__}: {count_parameters(model) / 1e6:.2f} M parameters')
    print(f'  input  {tuple(example.shape)}')
    for name, value in outputs.items():
        if hasattr(value, 'shape'):
            print(f'  {name:16s} {tuple(value.shape)}')
