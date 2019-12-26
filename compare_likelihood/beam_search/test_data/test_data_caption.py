import sys

sys.path.append('/home/mlspeech/gshalev/anaconda3/envs/python3_env/lib')
sys.path.append('/home/mlspeech/gshalev/gal/image_cap')


from language_model.word_language_model.data_holder import Corpus
from language_model.word_language_model.get_sentence_prop_by_trained_lm import get_sen_prop
from dataset_loader.datasets import CaptionDataset
from decoding_strategist_visualizations.decoding_strategist_utils import *
from decoding_strategist_visualizations.beam_search.beam_search_pack_utils import *
import torch
import torchvision.transforms as transforms


args = get_args()

def load_lm_model():
    torch.manual_seed(1111)

    p = '../../../language_model/word_language_model/model.pt'
    model = torch.load(p, map_location=torch.device(device))
    model.eval()

    return model

def visualize_att(seq, rev_word_map, top_seq_total_scors):
    words = [rev_word_map[ind] for ind in seq]
    lm_prop = get_sen_prop(words, lm_model, corpus, device)
    return words, lm_prop, sum(top_seq_total_scors[1:])


def run(encoder, decoder, word_map, rev_word_map, save_dir, image, image_name):
    seq, alphas, top_seq_total_scors, seq_sum, logits_list = beam_search_decode(encoder, image, args.beam_size,
                                                                                word_map, decoder)

    words, lm_prop, ic_prop = visualize_att(seq, rev_word_map, top_seq_total_scors)

    f = open(os.path.join(save_dir, 'test_seq_sum.txt'), 'a+')
    f.write('ic_prop: {}    LM Prop: {}     for: {}\n'.format(ic_prop, sum([x[1] for x in lm_prop]), image_name))


if __name__ == '__main__':
    global desktop_path, lm_model, corpus

    save_dir_name = 'likelihood_{}_{}'.format(args.beam_size, args.save_dir_name)
    model_path, save_dir = get_model_path_and_save_path(args, save_dir_name)

    encoder, decoder = get_models(model_path)

    word_map, rev_word_map = get_word_map(run_local=args.run_local)
    p = '/yoav_stg/gshalev/image_captioning/output_folder'
    coco_data = '../../../output_folder' if args.run_local else p

    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(coco_data, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
        batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
    lm_model = load_lm_model()
    corpus = Corpus('../../../language_model/word_language_model/data_dir')

    for i, (image, caps, caplens, allcaps) in enumerate(test_loader):
        if args.run_local and i > 20:
            break
        caps_ = [rev_word_map[x.item()] for x in caps[0]]
        ind = caps_.index('<end>')
        name = ' '.join(caps_[1:ind])
        image = image.to(device)
        run(encoder, decoder, word_map, rev_word_map, save_dir, image, name)

# test_data_caption.py