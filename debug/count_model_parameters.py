
import torch

# from standart_training.V_fixed_models_no_attention import DecoderWithoutAttention, Encoder
from standart_training.V_models_with_attention import Encoder, DecoderWithAttention
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()


def get_models(model_path, device):

    checkpoint = torch.load(model_path, map_location=torch.device(device))

    decoder = DecoderWithAttention(attention_dim=512,
    # decoder = DecoderWithoutAttention(attention_dim=512,
                                      embed_dim=512,
                                      decoder_dim=512,
                                      vocab_size=9490,
                                      device=device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder = decoder.to(device)
    decoder.eval()

    encoder = Encoder()
    encoder.load_state_dict(checkpoint['encoder'])
    encoder = encoder.to(device)
    encoder.eval()

    representations = checkpoint['representations']

    return encoder, decoder, representations

model_tar_file = 'NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
model_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(args.model, model_tar_file)

encoder, decoder, representations = get_models(model_path=model_path, device='cpu')
model_parameters =  sum(p.numel() for p in decoder.parameters() if p.requires_grad)
print('model {} has {} parameters'.format(args.model,model_parameters ))

