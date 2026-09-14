from brepdgt.datasets.abc_dataset import (
    ABCDataset,
    DeepCadDataset,
    ABC50,
    DeepCad30Dataset,
)

DATASETS = {
    ABCDataset.name: ABCDataset,
    DeepCadDataset.name: DeepCadDataset,
    ABC50.name: ABC50,
    DeepCad30Dataset.name: DeepCad30Dataset,
}
