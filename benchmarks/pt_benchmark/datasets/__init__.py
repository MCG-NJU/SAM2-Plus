from .custom import CustomDataset
from .badja import BADJADataset
from .trackinganygranularity import TrackingAnyGranularityDataset

def construct_dataset(dataset_name, dataset_root):
    """
    Constructs a dataset based on the given dataset name and root directory.

    Args:
        dataset_name (str): The name of the dataset.
        dataset_root (str): The root directory of the dataset.

    Returns:
        Dataset: The constructed dataset object.

    Raises:
        ValueError: If the dataset name is unknown.
    """
    if dataset_name == "custom":
        return CustomDataset(data_dir=dataset_root)
    elif dataset_name == "BADJA":
        return BADJADataset(data_dir=dataset_root)
    elif dataset_name == "tracking_any_granularity_val":
        return TrackingAnyGranularityDataset(data_dir=dataset_root, split="valid")
    elif dataset_name == "tracking_any_granularity_test":
        return TrackingAnyGranularityDataset(data_dir=dataset_root, split="test")
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))