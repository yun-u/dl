import importlib
import os
from contextlib import contextmanager
from datetime import datetime
from os import PathLike
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


@contextmanager
def working_directory(path: Union[str, PathLike]):
    """Context manager that temporarily changes the working directory to the specified 'path'.

    Args:
        path (`Union[str, PathLike]`): The path to the directory to set as the new working directory.

    Raises:
        OSError: If there's an issue changing the working directory.

    Examples:
        ```python
        with working_directory('/path/to/directory'):
            # Code executed inside this block will have '/path/to/directory' as the working directory.
        # After the block, the working directory is restored to its original state.
        ```
    """
    torch.nn.Linear
    old_path = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_path)


def import_from(name: str) -> Any:
    """Imports and returns an object based on the provided name.

    Args:
        name (`str`): The name of the object to import.

    Returns:
        `Any`: The imported object.

    Raises:
        AssertionError: If the provided name is not in the correct format.
    """
    split_name = name.rsplit(".", maxsplit=1)
    assert len(split_name) == 2, "Invalid name format: {name}"
    module_path, obj_name = split_name

    module = importlib.import_module(module_path)
    return getattr(module, obj_name)


def get_object(
    name: str, globals: Optional[Dict] = None, locals: Optional[Dict] = None
) -> Any:
    """Retrieves and returns an object by name from the given globals and locals dictionaries.

    Args:
        name (`str`): The name of the object to retrieve.
        globals (`Dict`): The dictionary of global variables.
        locals (`Dict`): The dictionary of local variables.

    Returns:
        `Any`: The retrieved object.
    """
    if locals and name in locals:
        return locals[name]
    if globals and name in globals:
        return globals[name]
    else:
        return import_from(name)


def get_datetime_string() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()
