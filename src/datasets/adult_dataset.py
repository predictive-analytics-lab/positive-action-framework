"""Functions for synthetic data."""
from typing import Tuple

from ethicml import Dataset, DataTuple, adult
from sklearn.preprocessing import MinMaxScaler


def adult_data(*, bin_nationality: bool) -> Tuple[Dataset, DataTuple]:
    """Generate very simple X data."""
    dataset = adult(binarize_nationality=bin_nationality)

    data = dataset.load()
    scaler = MinMaxScaler()
    scaler.fit(data.x[dataset.continuous_features])
    data = data.replace(x=data.x[dataset.continuous_features])

    return dataset, dataset.load()
