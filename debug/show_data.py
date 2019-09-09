import torch
import matplotlib.pyplot as plt
import os

desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
for i in range(5):
    dir = os.path.join(desktop_path, 'OOD_GIFs/data_ood_num_{}'.format(i))
    L = torch.load(dir)
    imgplot = plt.imshow(L['fig'])
    print(L['seq_sum'])
    plt.show()

desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
dir = os.path.join(desktop_path, 'GIFs')
for i, p in enumerate(os.listdir(dir)):
    L = torch.load(os.path.join(dir, p))
    print(L['seq_sum'])
    imgplot = plt.imshow(L['fig'])
    plt.show()

