## Install
You need [poetry](https://python-poetry.org/docs/#installation).
1. Create a new envronment
2. Install [PyTorch](https://pytorch.org/get-started/locally/#start-locally) - this depends on your machine, so you need to choose the right one for you.
3. `poetry install`
4. `poetry build`
5. `poetry install` (again)

The last 2 steps are for adding a shortcut so that you can run the model with `run`

## Run
### Run with default settings
`run`

### Run with modifications to the default dataset
`run data.batch_size=128`

### Change the dataset
`run data/schema=adult data=adu`


