"""Functions for synthetic data."""
from typing import Tuple

from ethicml import Dataset, DataTuple, adult


def adult_data(*, bin_nationality: bool) -> Tuple[Dataset, DataTuple]:
    """Generate very simple X data."""
    dataset = adult(binarize_nationality=bin_nationality)
    return dataset, dataset.load()
