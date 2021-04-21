"""Functions for synthetic data."""
from dataclasses import replace
from typing import Tuple

from ethicml import Dataset, DataTuple, adult, compas, credit
from ethicml.data.util import simple_spec


def adult_data(*, sens: str, bin_nationality: bool, bin_race: bool) -> Tuple[Dataset, DataTuple]:
    """Get the Audlt dataset."""
    if sens == "Binary-Married":
        dataset = adult(
            split="Custom", binarize_nationality=bin_nationality, binarize_race=bin_race
        )
        # discrete_features = reduce_feature_group(
        #     disc_feature_groups=dataset.disc_feature_groups,
        #     feature_group="marital-status",
        #     to_keep=simple_spec({sens: [
        #         "marital-status_Married-AF-spouse",
        #         "marital-status_Married-civ-spouse",
        #         "marital-status_Married-spouse-absent",
        #     ]}),
        #     remaining_feature_name="_not_Married",
        # )
        dataset = replace(
            dataset,
            sens_attr_spec=simple_spec(
                {
                    "marital-status": [
                        "marital-status_Married-AF-spouse",
                        "marital-status_Married-civ-spouse",
                        "marital-status_Married-spouse-absent",
                        "marital-status_Divorced",
                        "marital-status_Never-married",
                        "marital-status_Separated",
                        "marital-status_Widowed",
                    ]
                }
            ),
            s_prefix=["marital-status"],
            class_label_spec="salary_>50K",
            class_label_prefix=["salary"],
        )

        dt = dataset.load()

        dt.s['marital-status'] = (dt.s['marital-status'] < 3).astype(int)

        datatuple = dt
    else:
        dataset = adult(split=sens, binarize_nationality=bin_nationality, binarize_race=bin_race)
        datatuple = dataset.load(ordered=True)
    return dataset, datatuple


def credit_data() -> Tuple[Dataset, DataTuple]:
    """Get the Credit dataset."""
    dataset = credit()
    return dataset, dataset.load(ordered=True)


def compas_data() -> Tuple[Dataset, DataTuple]:
    """Get the Compas dataset."""
    dataset = compas()
    return dataset, dataset.load(ordered=True)
