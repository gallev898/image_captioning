import torchvision

from dataset_loader.custom_dataloader import Custom_Image_Dataset
from dataset_loader.flickr_parser import Flickr30k
from utils import *


def cifar10_loader(run_local, batch_size, num_workers):
    if run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        path = os.path.join(desktop_path, 'datasets/cifar10')
    else:
        path = '/yoav_stg/gshalev/semantic_labeling/cifar10'

    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Resize(256),
        data_normalization
        ])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root=path, train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def cifar100_loader(run_local, batch_size, num_workers):
    if run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        path = os.path.join(desktop_path, 'datasets/cifar100')
    else:
        path = '/yoav_stg/gshalev/semantic_labeling/cifar100'

    transform = transforms.Compose(
        [transforms.ToTensor(),
         data_normalization
         ])

    trainset = torchvision.datasets.CIFAR100(root=path, train=True,
                                             download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR100(root=path, train=False,
                                            download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader, classes


def svhn_loader(run_local, batch_size, num_workers):
    if run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        path = os.path.join(desktop_path, 'datasets/svhn')
    else:
        path = '/yoav_stg/gshalev/semantic_labeling/svhn'

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        data_normalization
        ])

    testset = torchvision.datasets.SVHN(root=path, split='test',
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return None, testloader, ['10', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def sbu_loader(run_local, batch_size, num_workers, crop=False):
    if run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        path = os.path.join(desktop_path, 'datasets/sbu')
    else:
        path = '/yoav_stg/gshalev/semantic_labeling/sbu'

    transform_test = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        data_normalization
        ])

    dataset = torchvision.datasets.SBU(root=path, transform=transform_test, target_transform=None, download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return dataloader


# def custom_loader(run_local, batch_size, num_workers):
#     desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
#     data_path = os.path.join(desktop_path, 'datasets/mscoco/val2014')
#
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         data_normalization
#         ])
#     train_dataset = Custom_Dataset(
#         root=data_path,
#         transform=transform
#         )
#     train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers,
#                                                shuffle=True)
#     return train_loader


def flicker_loader(run_local):
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    data_path = os.path.join(desktop_path, 'datasets/flicker') if run_local else os.curdir


    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        data_normalization
        ])

    train_loader = Flickr30k(root=os.path.join(data_path, 'Flicker8k_Dataset'),
                                                 ann_file=os.path.join(data_path, 'Flickr8k_text/Flickr8k.token.txt'), transform=transform)



    return train_loader


def load(dataset, run_local, batch_size, num_workers):
    if dataset == 'cifar10':
        return cifar10_loader(run_local, batch_size, num_workers)
    elif dataset == 'cifar100':
        return cifar100_loader(run_local, batch_size, num_workers)
    elif dataset == 'svhn':
        return svhn_loader(run_local, batch_size, num_workers)
    elif dataset == 'sbu':
        return sbu_loader(run_local, batch_size, num_workers)
    # elif dataset == 'custom':
        # return custom_loader(run_local, batch_size, num_workers)
    elif dataset == 'flicker':
        return flicker_loader(run_local)
# dataloader.py
