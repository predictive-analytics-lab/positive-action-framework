"""How would a perfect predictor perform?"""

from ethicml import DataTuple, InAlgorithm, Prediction, implements


class Oracle(InAlgorithm):
    """A perfect predictor."""

    @implements(InAlgorithm)
    def __init__(self) -> None:
        super().__init__(name="Oracle", is_fairness_algo=False)

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: DataTuple) -> Prediction:
        return Prediction(hard=test.y[test.y.columns[0]])
