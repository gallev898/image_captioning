import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage.transform


def visualization(image, alphas, words, pos, top_seq_total_scors, top_seq_total_scors_exp, smooth, save_dir,
                  image_name):
    for t in range(len(words)):
        if t > 50:
            break
        plt.subplot(np.ceil(len(words) / 5.), 5, t + 1)

        plt.text(0, 1, '{}\n{}\n {:.4f} - \n {:.4f}'.format(words[t], pos[t], top_seq_total_scors[t],
                                                            top_seq_total_scors_exp[t]),
                 color='black', backgroundcolor='white',
                 fontsize=12)
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

    plt.savefig(os.path.join(save_dir, 'temp_{}'.format(image_name)))
    plt.clf()
    return words, image