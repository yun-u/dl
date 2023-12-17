import logging
import subprocess
from os import PathLike
from typing import Union

logger = logging.getLogger(__name__)


def is_git_repository(directory: Union[str, PathLike]) -> bool:
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=directory,
            stderr=subprocess.STDOUT,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def add_and_commit_changes(commit_message: str) -> None:
    try:
        subprocess.run(["git", "add", "."])
        subprocess.run(["git", "commit", "-m", commit_message])
    except subprocess.CalledProcessError as e:
        logging.error(f"Error adding and commiting changes: {e}")
