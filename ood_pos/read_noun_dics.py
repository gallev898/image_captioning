import torch
import numpy as np


def load_dics():
    noun_dics = torch.load('TRAIN_NOUN_dics')

    flickr_dic = noun_dics['flickr_dic']
    in_dis_dic = noun_dics['in_dis_dic']

    flickr_values = list(flickr_dic.values())
    in_dis_values = list(in_dis_dic.values())

    flickr_mean_val = np.mean(flickr_values)
    in_dis_mean_val = np.mean(in_dis_values)

    flickr_keys = list(flickr_dic.keys())
    in_dis_keys = list(in_dis_dic.keys())

    return flickr_dic, in_dis_dic, flickr_values, in_dis_values, flickr_mean_val, in_dis_mean_val, flickr_keys, in_dis_keys


if __name__ == '__main__':
    flickr_dic, in_dis_dic, flickr_values, in_dis_values, flickr_mean_val, in_dis_mean_val, flickr_keys, in_dis_keys = load_dics()

    # Trim uncomon words from flickr dic
    deleted_keys = 0
    size_of_flicker_dic_befor_triming = len(flickr_keys)
    for key in flickr_keys:
        if flickr_dic[key] < flickr_mean_val:
            deleted_keys += 1
            print('{}   {}'.format(key, flickr_dic[key]))
            flickr_dic.pop(key)

    size_of_flicker_dic_after_triming = len(flickr_dic)
    assert size_of_flicker_dic_after_triming == size_of_flicker_dic_befor_triming - deleted_keys

    # Remove from flickr words that apper in in_dis
    new_flickr_keys = list(flickr_dic.keys())
    deleted_keys = 0
    size_of_flicker_dic_befor_triming = len(new_flickr_keys)

    for key in new_flickr_keys:
        if key in in_dis_keys:
            deleted_keys += 1
            print('{}   {}'.format(key, flickr_dic[key]))
            flickr_dic.pop(key)
    size_of_flicker_dic_after_triming = len(flickr_dic)
    assert size_of_flicker_dic_after_triming == size_of_flicker_dic_befor_triming - deleted_keys
    print(list(flickr_dic.keys()))

    g = 9
