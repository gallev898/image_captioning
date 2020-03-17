import argparse
import os
import torch

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform


def get_args():
    # args
    parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption')
    parser.add_argument('--model', type=str)
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--limit_ex', type=int, default=5)
    parser.add_argument('--beam_size', default=1, type=int)
    parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')
    args = parser.parse_args()

    return args

# global
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_name = 'coco_5_cap_per_img_5_min_word_freq'
filename = 'BEST_checkpoint_' + data_name + '.pth.tar'

def visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                  image_name):
    str_dis = ""
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '{}'.format(words[t]))
        str_dis += '{} {} {}\n'.format(words[t], pos[t], top_seq_total_scors_exp[t])
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig(os.path.join(save_dir, image_name))
    plt.clf()
    with open(os.path.join(save_dir,"{}.txt".format(image_name)), "w") as text_file:
        text_file.write(str_dis)
    return words, image
"""
def visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                  image_name):
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '{}'.format(words[t],
        # plt.text(0, 1, '{}\n{}\n {:.4f}  \n {:.4f}'.format(words[t], pos[t], top_seq_total_scors[t],
                                                            ),
                                                            # top_seq_total_scors_exp[t]),
                 color='black', backgroundcolor='white',
                 fontsize=10)
        plt.imshow(image)
        current_alpha = alphas[t, :]
        if smooth:
            alpha = skimage.transform.pyramid_expand(current_alpha.numpy(), upscale=24, sigma=8)
        else:
            alpha = skimage.transform.resize(current_alpha.numpy(), [14 * 24, 14 * 24])
        if t == 0:
            plt.imshow(alpha, alpha=0)
        else:
            plt.imshow(alpha, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig(os.path.join(save_dir, image_name))
    plt.clf()
    return words, image
"""