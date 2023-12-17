import logging
from pathlib import Path

from dl.utils.git import is_git_repository


def test_is_git_repository():
    logging.info(is_git_repository(Path().cwd()))
