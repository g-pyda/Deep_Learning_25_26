# data loading and preprocessing

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from utils import error_logger, basic_logger, global_log

@basic_logger(global_log)
@error_logger(global_log.error_path)
def get_data(config):
    # getting the data from configuration
    path = config["path"]
    data_transforms = {
        "train_transforms": [],
        "test_transforms": []
    }

    for key, trlist in data_transforms:
        if config[key]:
            for trans in config[key]:
                match (trans["type"]):
                    case "RandomRotation":
                        trlist.append(transforms.RandomRotation(trans["value"]))
                    case "RandomHorizontalFlip":
                        trlist.append(transforms.RandomHorizontalFlip(trans["value"]))
                    case "RandomCenterCrop":
                        trlist.append(transforms.RandomCenterCrop(trans["value"]))
                    case "RandomResize":
                        trlist.append(transforms.RandomResize(trans["value"]))
                    case "Normalize":
                        trlist.append(transforms.Normalize(*trans["value"]))
                    case "ToTensor":
                        trlist.append(transforms.ToTensor())


    # applying the transforms and creating the dataset
    train_transform = transforms.Compose(data_transforms["train_transforms"])
    test_transform = transforms.Compose(data_transforms["test_transforms"])

    train_dataset = datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=test_transform
    )

    return train_dataset, test_dataset

def load_data(dataset, subset_size=None, batch_size=32, shuffle=False, num_workers=4):
    # generating the subset if needed
    subset = dataset
    if subset_size:
        subset = Subset(dataset, range(subset_size))

    # creating the dataloader
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return loader