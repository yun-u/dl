import re
from collections import defaultdict
from typing import (
    Dict,
    Iterable,
    List,
    TypeVar,
    Union,
)

K = TypeVar("K")
V = TypeVar("V")


def transpose(
    inputs: Union[List[Dict[K, V]], Dict[K, List[V]]]
) -> Union[Dict[K, List[V]], List[Dict[K, V]]]:
    """Transpose a list of dictionaries or a dictionary of lists.

    Args:
        inputs (`Union[List[Dict[K, V]], Dict[K, List[V]]]`): The input data to transpose.

    Returns:
        `Union[Dict[K, List[V]], List[Dict[K, V]]]`: The transposed data.

    Raises:
        ValueError: If the input format is invalid or lengths of input values are not equal.
    """
    if isinstance(inputs, list):
        if not isinstance(inputs[0], dict):
            raise ValueError("The input elements in the list should be dictionaries.")

        outputs = defaultdict(list)
        for input in inputs:
            for key, value in input.items():
                outputs[key].append(value)

        return dict(outputs)
    elif isinstance(inputs, dict):
        if len(set((len(v) for v in inputs.values()))) > 1:
            raise ValueError("The lengths of the input values are not all equal.")

        outputs = []
        keys = inputs.keys()
        for values in zip(*inputs.values()):
            outputs.append({key: value for key, value in zip(keys, values)})
        return outputs

    raise ValueError(
        "The input should be either a list of dictionaries or a dictionary."
    )


def matched_groups(iterable: Iterable, pattern: str):
    """Find and yield matched groups in the iterable based on a regular expression pattern.

    Args:
        iterable (`Iterable`): The input iterable.
        pattern (`str`): The regular expression pattern to match.

    Yields:
        `Tuple`: Matched groups as tuples.
    """
    p = re.compile(pattern)
    matches = filter(None, map(lambda x: p.match(x), iterable))
    yield from map(lambda x: x.groups(), matches)
