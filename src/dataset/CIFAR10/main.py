
from torchvision.datasets import CIFAR10
from typing import Optional,Callable

class CIFAR10Dataset(CIFAR10):
    def __init__(self, root: str = "./dataset", train: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False) -> None:
        super().__init__(root=root,train=train,transform=transform,download=download)
        