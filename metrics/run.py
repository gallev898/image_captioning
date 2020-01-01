import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from metrics.calc_metric import bhatta_dist, auroc, auprIn, auprOut


if __name__ == '__main__':
    model_name = 'run_8'
    dics_path = '../pos_histograms/create_dic_for_histograms/pos_dic/{}/pos_dic'.format(model_name)

    custom_dics = set()
    test_dics = set()
    random_dics = set()
    for d in os.listdir(dics_path):
        if 'random' in d:
            random_dics.add(d)
            continue
        if 'test' in d:
            test_dics.add(d)
            continue
        if 'custom' in d:
            custom_dics.add(d)


    custom_dics = sorted(custom_dics)
    test_dics = sorted(test_dics)
    random_dics = sorted(random_dics)

    # section: custom VS test
    for c, t in zip(custom_dics, test_dics):

        c_dic = torch.load('{}/{}'.format(dics_path, c))
        c_sentence_likelihood = c_dic['sentence_likelihood']
        c_sentence_likelihood = [i for i in c_sentence_likelihood if i != None]
        # c_sentence_likelihood = np.exp(c_sentence_likelihood)

        t_dic = torch.load('{}/{}'.format(dics_path, t))
        t_sentence_likelihood = t_dic['sentence_likelihood']
        t_sentence_likelihood = [i for i in t_sentence_likelihood if i != None]
        # t_sentence_likelihood = np.exp(t_sentence_likelihood)

        bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
        auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
        auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
        auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)

        print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t,c,bhatta_score, auroc_score[0], auprIn_score, auprOut_score))

        minr, maxr = min(c_sentence_likelihood+t_sentence_likelihood), max(c_sentence_likelihood+t_sentence_likelihood)
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.hist(c_sentence_likelihood, density=True, stacked=True, bins=np.arange(minr, maxr, (maxr-minr)/float(60)), color='red', alpha=0.4)
        plt.hist(t_sentence_likelihood, density=True, stacked=True, bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
                 color='blue', alpha=0.4)
        plt.xlabel("sentence log probability", fontsize=10)
        plt.ylabel("probability", fontsize=10)
        plt.savefig('{}.png'.format(c))
        plt.clf()

    print('====================================================================')
    # section: custom VS test
    for c, t in zip(random_dics, test_dics):

        c_dic = torch.load('{}/{}'.format(dics_path, c))
        c_sentence_likelihood = c_dic['sentence_likelihood']
        c_sentence_likelihood = [i for i in c_sentence_likelihood if i != None]
        # c_sentence_likelihood = np.exp(c_sentence_likelihood)

        t_dic = torch.load('{}/{}'.format(dics_path, t))
        t_sentence_likelihood = t_dic['sentence_likelihood']
        t_sentence_likelihood = [i for i in t_sentence_likelihood if i != None]
        # t_sentence_likelihood = np.exp(t_sentence_likelihood)

        bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
        auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
        auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
        auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)

        print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t,c,bhatta_score, auroc_score[0], auprIn_score, auprOut_score))

        minr, maxr = min(c_sentence_likelihood+t_sentence_likelihood), max(c_sentence_likelihood+t_sentence_likelihood)
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.hist(c_sentence_likelihood, density=True, stacked=True, bins=np.arange(minr, maxr, (maxr-minr)/float(60)), color='red', alpha=0.4)
        plt.hist(t_sentence_likelihood, density=True, stacked=True, bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
                 color='blue', alpha=0.4)
        plt.xlabel("sentence log probability", fontsize=10)
        plt.ylabel("probability", fontsize=10)
        plt.savefig('{}.png'.format(c))
        plt.clf()