"""Utility functions."""
import collections
import warnings
from typing import Any, Dict, List, MutableMapping

import pandas as pd
from ethicml import Prediction

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def flatten(d: MutableMapping[Any, Any], parent_key: str = "", sep: str = ".") -> Dict[Any, Any]:
    """Flatten a nested dictionary by separating the keys with `sep`."""
    items: List[Any] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def facct_mapper(facct_out: Prediction) -> Prediction:
    """Map from groups to outcomes."""
    lookup = {
        0: 5,
        1: 6,
        2: 3,
        3: 4,
        4: 7,
        5: 7,
        6: 3,
        7: 4,
        8: 7,
        9: 7,
        10: 1,
        11: 4,
        12: 7,
        13: 7,
        14: 1,
        15: 2,
        16: 8,
        17: 7,
        18: 8,
        19: 8,
        20: 8,
        21: 7,
        22: 8,
        23: 8,
        24: 8,
        25: 7,
        26: 8,
        27: 8,
        28: 8,
        29: 7,
        30: 1,
        31: 2,
    }

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def facct_mapper_2(facct_out: Prediction) -> Prediction:
    """Map from groups to outcomes."""
    lookup = {-1: 0, 1: 1, 2: 1, 3: 2, 4: 1, 5: 0, 6: 0, 7: 0, 8: 1}

    preds = pd.Series({i: lookup[d] for i, d in enumerate(facct_out.hard)})

    return Prediction(hard=preds, info=facct_out.info)


def facct_mapper_outcomes(mapped: Prediction, fair: bool) -> Prediction:
    """Make the final outcome."""
    lookup = {0: 0, 1: 1, 2: 1 if fair else 0}

    preds = pd.Series({i: lookup[d] for i, d in enumerate(mapped.hard)})

    return Prediction(hard=preds, info=mapped.info)