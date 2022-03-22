


from torchvision.datasets import MNIST

class MNISTDataset(MNIST):
    def __init__(self,root='./dataset',train=True,transform=None):
        super().__init__(root,transform,train,download=True)
    