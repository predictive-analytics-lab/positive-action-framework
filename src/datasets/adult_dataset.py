"""Functions for synthetic data."""
from typing import Tuple

from ethicml import Dataset, DataTuple, adult


def adult_data(*, bin_nationality: bool, bin_race: bool) -> Tuple[Dataset, DataTuple]:
    """Generate very simple X data."""
    dataset = adult(binarize_nationality=bin_nationality, binarize_race=bin_race)
    return dataset, dataset.load(ordered=True)
