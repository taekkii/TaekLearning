

from torchvision.datasets import MNIST
from torchvision.datasets import CIFAR10


class DatasetNotExistError(Exception):
    pass




dataset_dict = dict(mnist=MNIST,
                    cifar10=CIFAR10)


def get( dataset_name:str , dataset_config:dict ):
    print(f"Trying to get dataset [{dataset_name}]...\n")

    #----- GUARD -----#
    if dataset_name.lower() not in dataset_dict:
        raise DatasetNotExistError(f"Unregistered dataset : [{dataset_name}]")
    
    dataset_name = dataset_name.lower()

    return dataset_dict[dataset_name](**dataset_config)


