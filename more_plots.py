from pathlib import Path

from matplotlib import pyplot as plt
import pandas as pd
import typer


def plotting(csv_path: Path):
    data = pd.read_csv(csv_path)

    data = data[[col for col in data.columns if "Table6_test" in col]]

    data = data.transpose()
    data = data.rename(
        index={
            "Table6_test/Ours/recon_l1 - feature math": "Math",
            "Table6_test/Ours/recon_l1 - feature physics": "Physics",
            "Table6_test/Ours/recon_l1 - feature biology": "Biology",
            "Table6_test/Ours/recon_l1 - feature history": "History",
            "Table6_test/Ours/recon_l1 - feature language": "Language",
            "Table6_test/Ours/recon_l1 - feature geography": "Geography",
            "Table6_test/Ours/recon_l1 - feature literature": "Literature",
            "Table6_test/Ours/recon_l1 - feature chemistry": "Chemistry",
            "Table6_test/Ours/recon_l1 - feature essay": "Essay",
        },
        columns={0: "CycleGAN 400 epochs", 1: "CycleGAN", 2: "PAF w/ CycleLoss", 3: "PAF"},
    )
    print(data.head())
    ax = data.plot(kind='bar', width=1, colormap='Accent')
    ax.legend(bbox_to_anchor=(1.0, 1.0))
    plt.savefig('admissions_recon', bbox_inches='tight')


if __name__ == '__main__':
    typer.run(plotting)
