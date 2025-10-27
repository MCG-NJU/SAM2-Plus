from .customdataset import CustomDataset
from .got10kdataset import GOT10KDataset
from .trackingnetdataset import TrackingNetDataset
from .trackinganygranularitydataset import TrackingAnyGranularityDataset

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
    if dataset_name == "customdataset":
        return CustomDataset(dataset_root)
    elif dataset_name == "tracking_any_granularity_val":
        return TrackingAnyGranularityDataset(dataset_root, 'valid')
    elif dataset_name == "tracking_any_granularity_test":
        return TrackingAnyGranularityDataset(dataset_root, 'test')
    elif dataset_name == "GOT-10k_test" or  dataset_name == "got10k_test":
        # print("GOT-10k dataset test ground truth not available. Using validation set.")
        return GOT10KDataset(dataset_root, split="test")
    elif dataset_name == "GOT-10k_val" or dataset_name == "got10k_val": 
        return GOT10KDataset(dataset_root, split="val")
    elif dataset_name == "TrackingNet" or dataset_name == "trackingnet":
        return TrackingNetDataset(dataset_root)
    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))