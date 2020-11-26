"""How would a perfect predictor perform?"""

import pandas as pd
from ethicml import (
    Accuracy,
    DataTuple,
    InAlgorithm,
    Prediction,
    ProbPos,
    adult,
    implements,
    metric_per_sensitive_attribute,
    train_test_split,
)


class DPOracle(InAlgorithm):
    """A perfect DP predictor."""

    @implements(InAlgorithm)
    def __init__(self, acceptance_rate: float) -> None:
        super().__init__(name="DP Oracle", is_fairness_algo=True)
        self.acceptance_rate = acceptance_rate

    @staticmethod
    def get_fraction(y_group: pd.Series, num_samples: int) -> pd.DataFrame:
        """Get num samples to support DP."""
        if y_group[y_group[y_group.columns[0]] == 1].count().item() >= num_samples:
            to_return = y_group[y_group[y_group.columns[0]] == 1].sample(n=num_samples)
        if y_group[y_group[y_group.columns[0]] == 1].count().item() < num_samples:
            _acc = y_group[y_group[y_group.columns[0]] == 1].sample(frac=1.0)
            to_get = num_samples - _acc.count().item()
            _samples = y_group[y_group[y_group.columns[0]] == 0].sample(n=to_get)
            _samples.values[:] = 1
            to_return = pd.concat([_acc, _samples])
        else:
            raise NotImplementedError("")
        return to_return

    @implements(InAlgorithm)
    def run(self, train: DataTuple, test: DataTuple) -> Prediction:
        preds = test.y.copy()
        preds.values[:] = 0

        for sens in test.s[test.s.columns[0]].unique():
            even_samples = int((test.s[test.s[test.s.columns[0]] == sens].count() * self.acceptance_rate))

            if test.s[test.s[test.s.columns[0]] == sens].count().item() < even_samples:
                raise RuntimeError("Acceptance Threshold too great. Not enough samples per group.")

            y_group = test.y[test.s[test.s.columns[0]] == sens]
            _preds = self.get_fraction(y_group, even_samples)
            preds.values[_preds.index] = 1

        return Prediction(hard=preds[preds.columns[0]])


if __name__ == '__main__':
    clf = DPOracle(0.25)

    train_data, test_data = train_test_split(adult().load())

    predictions = clf.run(train_data, test_data)
    print(f"Acc: {Accuracy().score(predictions, test_data)}")
    print(f"DP: {ProbPos().score(predictions, test_data)}")
    print(f"DP: {metric_per_sensitive_attribute(predictions, test_data, ProbPos())}")
