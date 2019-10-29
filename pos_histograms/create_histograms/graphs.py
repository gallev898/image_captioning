import matplotlib.pyplot as plt
import torch
import numpy as np
import os

if __name__ == '__main__':
    dic_type = 'random'

    if not os.path.exists(dic_type):
        os.mkdir(dic_type)

    dic = torch.load('../create_dic/pos_dic/pos_dic_{}'.format(dic_type))
    pos_dic = dic['pos']
    noun_phrase_sum_of_log_prop = dic['noun_phrase_sum_of_log_prop']


    plt.clf()

    _ = plt.hist(noun_phrase_sum_of_log_prop, normed=True, bins=np.arange(min(noun_phrase_sum_of_log_prop), max(noun_phrase_sum_of_log_prop)+0.1, 0.01), color='purple')
    plt.xlabel('noun_phrase_sum_of_log_prop')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_xlabel()))

    avg = {}
    for k in pos_dic.keys():

        avg[k] = {'exp': np.average(pos_dic[k]['exp']),
                  'logits': np.average(pos_dic[k]['logits']),
                  'alphas_max': np.average(pos_dic[k]['alphas_max']),
                  'alphas_var': np.average(pos_dic[k]['alphas_var']),
                  'count': len(pos_dic[k]['exp'])
                  }
        plt.clf()
        _ = plt.hist(pos_dic[k]['exp'], normed=True, bins=np.arange(0.0, 1.1, 0.01), color='blue')
        plt.xlabel(k + ' exp')
        # plt.show()
        plt.savefig(os.path.join(dic_type, plt.gca().get_xlabel()))

        plt.clf()
        min_logit = min(pos_dic[k]['logits'])
        max_logit = max(pos_dic[k]['logits'])
        _ = plt.hist(pos_dic[k]['logits'], normed=True, bins=np.arange(min_logit, max_logit + 0.1, 0.1), color='red')
        plt.xlabel(k + ' logits')
        # plt.show()
        plt.savefig(os.path.join(dic_type, plt.gca().get_xlabel()))

    plt.clf()
    avg_exp = [avg[k]['exp'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), avg_exp, color='pink')
    plt.title('exp avg')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_title()))

    plt.clf()
    avg_logits = [avg[k]['logits'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), avg_logits, color='yellow')
    plt.title('logits avg')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_title()))

    plt.clf()
    count = [avg[k]['count'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), count, color='green')
    plt.title('count')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_title()))

    plt.clf()
    alphas = [avg[k]['alphas_max'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), alphas, color='orange')
    plt.title('alphas max avg')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_title()))

    plt.clf()
    alphas = [avg[k]['alphas_var'] for k in list(avg.keys())]
    plt.bar(list(avg.keys()), alphas, color='brown')
    plt.title('alphas var avg')
    # plt.show()
    plt.savefig(os.path.join(dic_type, plt.gca().get_title()))

    d = 0
