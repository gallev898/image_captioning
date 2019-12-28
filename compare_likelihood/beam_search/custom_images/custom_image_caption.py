import sys


sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap2')

from tqdm import tqdm
from dataset_loader.dataloader import load
from language_model.word_language_model.data_holder import Corpus
from language_model.word_language_model.get_sentence_prop_by_trained_lm import get_sen_prop
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *
import torch
from PIL import Image


args = get_args()


def load_lm_model():
    torch.manual_seed(1111)

    p = '../../../language_model/word_language_model/model.pt'
    model = torch.load(p, map_location=torch.device(device))
    model.eval()

    return model


def visualize_att(seq, rev_word_map, top_seq_total_scors):
    words = [rev_word_map[ind] for ind in seq]  # notice
    lm_likelihood = get_sen_prop(words, lm_model, corpus, device)
    return words, lm_likelihood, sum(top_seq_total_scors[1:])


def run(encoder, decoder, word_map, rev_word_map, save_dir, image, image_title):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, args.beam_size,
                                                                                word_map, decoder)

    words, lm_likelihood, ic_likelihood = visualize_att(seq, rev_word_map, top_seq_total_scors)

    f = open(os.path.join(save_dir, 'custom_seq_sum.txt'), 'a+')
    f.write('ic_likelihood: {}    LM Prop: {}     for: {}\n'.format(ic_likelihood, sum([x[1] for x in lm_likelihood]),
                                                                    image_title))


if __name__ == '__main__':
    global desktop_path, lm_model, corpus

    save_dir_name = 'likelihood_{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)
    encoder, decoder = get_models(model_path)
    word_map, rev_word_map = get_word_map(run_local=args.run_local)

    transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.ToTensor(),
        data_normalization
        ])


    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    lm_model = load_lm_model()
    corpus = Corpus('../../../language_model/word_language_model/data_dir')

    # image_name = 'Fork.png'
    ############
    dataloader = load('custom', args.run_local, 1, 1)

    for i, data in tqdm(enumerate(dataloader)):
        image = data[0].to(device)
        if image.shape[1] != 3:
            continue
        run(encoder, decoder, word_map, rev_word_map, save_dir, image, data[1][0])
    #########
    #
    # for image_name in os.listdir(os.path.join(desktop_path, 'custom_images')):
    #     image_path = 'custom_images/{}'.format(image_name)
    #     img = Image.open(os.path.join(desktop_path, image_path))
    #
    #     try:
    #         image = transform(img).unsqueeze(0)
    #     except:
    #         continue
    #
    #     run(encoder, decoder, word_map, rev_word_map, save_dir, image, image_name)


    # custom_image_caption.py