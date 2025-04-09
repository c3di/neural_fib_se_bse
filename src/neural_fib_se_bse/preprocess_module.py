"""Example module that adds two integers in C++."""

import os
import typing

print( "importing preprocess_module" )

if not typing.TYPE_CHECKING and os.getenv("PYBIND11_PROJECT_PYTHON_DEBUG"):
    print( "importing debug package" )
    from ._preprocess_module_d import *  # noqa: F403
    from ._preprocess_module_d import __version__  # noqa: F401, RUF100
else:
    print( "importing release package" )
    from ._preprocess_module import *  # noqa: F403
    from ._preprocess_module import __version__  # noqa: F401, RUF100
