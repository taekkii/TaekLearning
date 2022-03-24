
import torchvision.transforms as transforms

RGB_MEAN = (0.491,0.482,0.447)
RGB_STD  = (0.247,0.243,0.261)

TRANSFORM_DICT = {
    'basic': transforms.ToTensor(),
    'basic_cifar_augment': transforms.Compose([
        transforms.RandomCrop(size=32,padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    'cifar_autoaugment':transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(policy=transforms.autoaugment.AutoAugmentPolicy.CIFAR10),
        transforms.ToTensor()
    ])
}