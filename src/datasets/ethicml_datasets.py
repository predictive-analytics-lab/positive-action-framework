"""Functions for synthetic data."""
from typing import Tuple

from ethicml import Dataset, DataTuple, adult, compas, credit


def adult_data(*, sens: str, bin_nationality: bool, bin_race: bool) -> Tuple[Dataset, DataTuple]:
    """Get the Audlt dataset."""
    dataset = adult(split=sens, binarize_nationality=bin_nationality, binarize_race=bin_race)
    return dataset, dataset.load(ordered=True)


def credit_data() -> Tuple[Dataset, DataTuple]:
    """Get the Credit dataset."""
    dataset = credit()
    return dataset, dataset.load(ordered=True)


def compas_data() -> Tuple[Dataset, DataTuple]:
    """Get the Compas dataset."""
    dataset = compas()
    return dataset, dataset.load(ordered=True)
