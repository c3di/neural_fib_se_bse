"""Example module that adds two integers in C++."""

import os
import typing

cuda_path = os.environ['CUDA_PATH']
os.add_dll_directory(cuda_path + "/bin" )

if not typing.TYPE_CHECKING and os.getenv("PYBIND11_PROJECT_PYTHON_DEBUG"):
    from ._preprocess_module_d import *  # noqa: F403
    from ._preprocess_module_d import __version__  # noqa: F401, RUF100
else:
    from ._preprocess_module import *  # noqa: F403
    from ._preprocess_module import __version__  # noqa: F401, RUF100
