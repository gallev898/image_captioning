import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib
from metrics_roc_and_more.calc_metric import bhatta_dist, auroc, auprIn, auprOut

style.use('seaborn-poster') #sets the size of the charts
style.use('ggplot')
matplotlib.rc('font', size=22)

if __name__ == '__main__':
    model_name = 'run_8'
    dics_path = '/Users/gallevshalev/Desktop/trained_models/{}/pos_dic'.format(model_name)

    custom_dics = set()
    test_dics = set()
    random_dics = set()
    cartoon_dics = set()
    cropped_dics = set()
    salt_dics = set()
    jpeg_dics = set()
    snow_dics = set()
    aug_cartoon_dics = set()
    for d in os.listdir(dics_path):
        if 'random' in d and 'pos' in d:
            random_dics.add(d)
            continue
        if 'test' in d and 'pos' in d:
            test_dics.add(d)
            continue
        if 'custom' in d and 'pos' in d:
            custom_dics.add(d)
        if 'dic_cartoon' in d and 'pos' in d:
            cartoon_dics.add(d)
        if 'cropped' in d and 'pos' in d:
            cropped_dics.add(d)
        if 'salt' in d and 'pos' in d:
            salt_dics.add(d)
        if 'jpeg' in d and 'pos' in d:
            jpeg_dics.add(d)
        if 'snow' in d and 'pos' in d:
            snow_dics.add(d)
        if 'aug_cartoon' in d and 'pos' in d:
            aug_cartoon_dics.add(d)


    custom_dics = sorted(custom_dics)
    test_dics = sorted(test_dics)
    random_dics = sorted(random_dics)
    cartoon_dics = sorted(cartoon_dics)
    cropped_dics = sorted(cropped_dics)
    jpeg_dics = sorted(jpeg_dics)
    salt_dics = sorted(salt_dics)
    snow_dics = sorted(snow_dics)
    aug_cartoon_dics = sorted(aug_cartoon_dics)

    sentences_likelihood_str = 'sentence_likelihood'
    print('====================================================================')
    # section: salt and pepper VS test
    for c, t in zip(salt_dics, test_dics):


        c_dic = torch.load('{}/{}'.format(dics_path, c))
        c_sentence_likelihood = c_dic[sentences_likelihood_str]
        if c.startswith('NEW'):
            c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
        # c_sentence_likelihood = np.exp(c_sentence_likelihood)

        t_dic = torch.load('{}/{}'.format(dics_path, t))
        t_sentence_likelihood = t_dic[sentences_likelihood_str]
        if t.startswith('NEW'):
            t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
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
        plt.hist(c_sentence_likelihood, label='salt-and-pepper noise', density=True,
                 bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
        plt.hist(t_sentence_likelihood, label='COCO', density=True,
                 bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
                 color='blue', alpha=0.4)
        plt.legend(fontsize=25)
        plt.xlabel("log probability", fontsize=25)
        plt.ylabel("density", fontsize=25)
        plt.savefig('pngs/{}.png'.format(c))
        plt.clf()


    # print('====================================================================')
    # # section: jpeg VS test
    # for c, t in zip(jpeg_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t,c,bhatta_score, auroc_score[0], auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood+t_sentence_likelihood), max(c_sentence_likelihood+t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='JPEG compression', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()
    #
    # print('====================================================================')
    #
    # # section: custom VS test
    # for c, t in zip(custom_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t,c,bhatta_score, auroc_score[0], auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood+t_sentence_likelihood), max(c_sentence_likelihood+t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='Unknown objects', density=True, bins=np.arange(minr, maxr, (maxr-minr)/float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True, bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()
    #
    # print('====================================================================')
    # # section: random VS test
    # for c, t in zip(random_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t,c,bhatta_score, auroc_score[0], auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood+t_sentence_likelihood), max(c_sentence_likelihood+t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='random noise', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()
    #
    # print('====================================================================')
    # # section: cartoon VS test
    #
    # for c, t in zip(cartoon_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t, c, bhatta_score, auroc_score[0],
    #                                                                             auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood + t_sentence_likelihood), max(
    #         c_sentence_likelihood + t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='Cartoon', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()
    #
    # print('====================================================================')
    # # section: cropped VS test
    # for c, t in zip(cropped_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t, c, bhatta_score, auroc_score[0],
    #                                                                             auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood + t_sentence_likelihood), max(
    #         c_sentence_likelihood + t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='Cropped unknown objects', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()

    # print('====================================================================')
    # # section: snow VS test
    # for c, t in zip(snow_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     try:
    #         c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     except:
    #         c_sentence_likelihood = c_dic[sentences_likelihood_str]
    #
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     t_sentence_likelihood = t_dic[sentences_likelihood_str]
    #     t_sentence_likelihood = [i for i in t_sentence_likelihood if i != None]
    #
    #     # t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t, c, bhatta_score, auroc_score[0],
    #                                                                             auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood + t_sentence_likelihood), max(
    #         c_sentence_likelihood + t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='Cropped unknown objects', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()
    #
    # print('====================================================================')
    # # section: aug_cartoon VS test
    # for c, t in zip(aug_cartoon_dics, test_dics):
    #
    #
    #     c_dic = torch.load('{}/{}'.format(dics_path, c))
    #     try:
    #         c_sentence_likelihood = c_dic['generated_sentences_likelihood']
    #     except:
    #         c_sentence_likelihood = c_dic[sentences_likelihood_str]
    #     c_sentence_likelihood = [i[1] for i in c_sentence_likelihood if i != None]
    #     # c_sentence_likelihood = np.exp(c_sentence_likelihood)
    #     t_dic = torch.load('{}/{}'.format(dics_path, t))
    #     try:
    #         t_sentence_likelihood = t_dic['generated_sentences_likelihood']
    #     except:
    #         t_sentence_likelihood = t_dic[sentences_likelihood_str]
    #
    #     try:
    #         t_sentence_likelihood = [i[1] for i in t_sentence_likelihood if i != None]
    #     except:
    #         t_sentence_likelihood = [i for i in t_sentence_likelihood if i != None]
    #     # t_sentence_likelihood = np.exp(t_sentence_likelihood)
    #     bhatta_score = bhatta_dist(t_sentence_likelihood, c_sentence_likelihood)
    #     auroc_score = auroc(t_sentence_likelihood, c_sentence_likelihood)
    #     auprIn_score = auprIn(t_sentence_likelihood, c_sentence_likelihood)
    #     auprOut_score = auprOut(t_sentence_likelihood, c_sentence_likelihood)
    #
    #     print('{} - {}:  bhatta: {}  auroc: {}  auprin: {}  auprout: {}'.format(t, c, bhatta_score, auroc_score[0],
    #                                                                             auprIn_score, auprOut_score))
    #
    #     minr, maxr = min(c_sentence_likelihood + t_sentence_likelihood), max(
    #         c_sentence_likelihood + t_sentence_likelihood)
    #     ax = plt.subplot(111)
    #     ax.spines["top"].set_visible(False)
    #     ax.spines["right"].set_visible(False)
    #     ax.get_xaxis().tick_bottom()
    #     ax.get_yaxis().tick_left()
    #     plt.hist(c_sentence_likelihood, label='Cropped unknown objects', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)), color='red', alpha=0.4)
    #     plt.hist(t_sentence_likelihood, label='COCO', density=True,
    #              bins=np.arange(minr, maxr, (maxr - minr) / float(60)),
    #              color='blue', alpha=0.4)
    #     plt.legend()
    #     plt.xlabel("caption log probability", fontsize=10)
    #     plt.ylabel("density", fontsize=10)
    #     plt.savefig('pngs/{}.png'.format(c))
    #     plt.clf()