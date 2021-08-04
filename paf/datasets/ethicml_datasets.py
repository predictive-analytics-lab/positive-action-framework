"""Functions for synthetic data."""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path

from ethicml import Dataset, DataTuple, adult, compas, credit
from ethicml.data.util import LabelSpec, flatten_dict, simple_spec

TGT_NAME = "salary_>50K"


def adult_data(*, sens: str, bin_nationality: bool, bin_race: bool) -> tuple[Dataset, DataTuple]:
    """Get the Audlt dataset."""
    if sens == "Binary-Married":
        dataset = adult(
            split="Custom", binarize_nationality=bin_nationality, binarize_race=bin_race
        )
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
            class_label_spec=TGT_NAME,
            class_label_prefix=["salary"],
        )

        dt = dataset.load()

        dt.s['marital-status'] = (dt.s['marital-status'] < 3).astype(int)

        datatuple = dt
    else:
        dataset = adult(split=sens, binarize_nationality=bin_nationality, binarize_race=bin_race)
        datatuple = dataset.load(ordered=True)
    return dataset, datatuple


def semi_adult_data(
    *, sens: str, bin_nationality: bool, bin_race: bool
) -> tuple[Dataset, DataTuple]:
    """Get the Audlt dataset."""
    disc_feature_groups = {
        "education": [
            "education_10th",
            "education_11th",
            "education_12th",
            "education_1st-4th",
            "education_5th-6th",
            "education_7th-8th",
            "education_9th",
            "education_Assoc-acdm",
            "education_Assoc-voc",
            "education_Bachelors",
            "education_Doctorate",
            "education_HS-grad",
            "education_Masters",
            "education_Preschool",
            "education_Prof-school",
            "education_Some-college",
        ],
        "marital-status": [
            "marital-status_Divorced",
            "marital-status_Married-AF-spouse",
            "marital-status_Married-civ-spouse",
            "marital-status_Married-spouse-absent",
            "marital-status_Never-married",
            "marital-status_Separated",
            "marital-status_Widowed",
        ],
        "native-country": [
            "native-country_United-States",
            "native-country_not-United-States",
        ],
        "race": [
            "race_Amer-Indian-Eskimo",
            "race_Asian-Pac-Islander",
            "race_Black",
            "race_Other",
            "race_White",
        ],
        "relationship": [
            "relationship_Husband",
            "relationship_Not-in-family",
            "relationship_Other-relative",
            "relationship_Own-child",
            "relationship_Unmarried",
            "relationship_Wife",
        ],
        "salary": ["salary_<=50K", TGT_NAME],
        "sex": ["sex_Female", "sex_Male"],
        "workclass": [
            "workclass_Federal-gov",
            "workclass_Local-gov",
            "workclass_Private",
            "workclass_Self-emp-inc",
            "workclass_Self-emp-not-inc",
            "workclass_State-gov",
            "workclass_Without-pay",
        ],
    }
    discrete_features = flatten_dict(disc_feature_groups)

    continuous_features = [
        "age",
        "capital-gain",
        "capital-loss",
        "education-num",
        "hours-per-week",
        "directness",
        "caring",
    ]

    sens_attr_spec: str | LabelSpec
    sens_attr_spec = "sex_Male"
    s_prefix = ["sex"]
    class_label_spec = TGT_NAME
    class_label_prefix = ["salary"]

    name = f"SemiSynthetic Adult"
    assert len(discrete_features) == 47  # 43 (discrete) features + 4 class labels

    dataset = Dataset(
        name=name,
        num_samples=45222,
        features=discrete_features + continuous_features,
        cont_features=continuous_features,
        sens_attr_spec=sens_attr_spec,
        class_label_spec=class_label_spec,
        filename_or_path=Path(__file__).parent / "csvs" / "adult_syn.csv",
        s_prefix=s_prefix,
        class_label_prefix=class_label_prefix,
        discrete_only=False,
        discrete_feature_groups=disc_feature_groups,
    )

    dt = dataset.load()
    datatuple = dt
    return dataset, datatuple


def credit_data() -> tuple[Dataset, DataTuple]:
    """Get the Credit dataset."""
    dataset = credit()
    return dataset, dataset.load(ordered=True)


def compas_data() -> tuple[Dataset, DataTuple]:
    """Get the Compas dataset."""
    dataset = compas()
    return dataset, dataset.load(ordered=True)
