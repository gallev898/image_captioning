import torch

from dataset_loader.datasets2 import CaptionDataset


train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_f, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
