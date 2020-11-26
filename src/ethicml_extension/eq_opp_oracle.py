"""How would a perfect predictor perform?"""
import numpy as np
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


class EqOppOracle(InAlgorithm):
    """A perfect DP predictor."""

    @implements(InAlgorithm)
    def __init__(self, acceptance_rate: float) -> None:
        super().__init__(name="DP Oracle", is_fairness_algo=True)
        self.acceptance_rate = acceptance_rate

    @staticmethod
    def get_fraction(y_group: pd.Series, num_samples: int) -> np.ndarray:
        """Get number of samples that support EqOpp."""
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
    def run(self, train_data: DataTuple, test_data: DataTuple) -> Prediction:
        assert train_data.y.shape[0] == test_data.y.shape[0], "Weird model - use train to sneak in data."

        preds = test_data.y.copy()
        preds.values[:] = 0

        for sens in test_data.s[test_data.s.columns[0]].unique():
            even_samples = int(
                (
                    test_data.s[
                        (train_data.y[train_data.y.columns[0]] == 1) & (test_data.s[test_data.s.columns[0]] == sens)
                    ].count()
                    * self.acceptance_rate
                )
            )

            if (
                test_data.s[
                    (train_data.y[train_data.y.columns[0]] == 1) & (test_data.s[test_data.s.columns[0]] == sens)
                ]
                .count()
                .item()
                < even_samples
            ):
                raise RuntimeError("Acceptance Threshold too great. Not enough samples per group.")

            y_group = test_data.y[
                (train_data.y[train_data.y.columns[0]] == 1) & (test_data.s[test_data.s.columns[0]] == sens)
            ]
            _preds = self.get_fraction(y_group, even_samples)
            preds.values[_preds.index] = 1

        return Prediction(hard=preds[preds.columns[0]])


if __name__ == '__main__':
    clf = EqOppOracle(0.25)
    train, test = train_test_split(adult().load())
    predictions = clf.run(
        test.replace(y=pd.DataFrame(np.random.binomial(1, 0.8, test.y.shape[0]), columns=["qw"])),
        test,
    )
    print(f"Acc: {Accuracy().score(predictions, test)}")
    print(f"DP: {ProbPos().score(predictions, test)}")
    print(f"DP: {metric_per_sensitive_attribute(predictions, test, ProbPos())}")
