import numpy as np

from dl.visualize.utils import display_text_confusion_matrix


def test_display_text_confusion_matrix():
    print()
    display_text_confusion_matrix([1, 0], [1, 0])
    display_text_confusion_matrix(
        np.random.randint(0, 10, (100,)), np.random.randint(0, 10, (100,))
    )
