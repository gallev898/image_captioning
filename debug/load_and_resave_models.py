import torch
import argparse
import os

# section: args
parser = argparse.ArgumentParser(description='')
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--model', type=str)
args = parser.parse_args()

# section: initialization
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar'
local_model_path = '/Users/gallevshalev/Desktop/trained_models/{}/{}'.format(args.model, model_name)
server_model_path = '/yoav_stg/gshalev/image_captioning/{}/{}'.format(args.model, model_name)

# section: load model
model_path = local_model_path if args.run_local else server_model_path
# saved_dict = torch.load(model_path)
saved_dict = torch.load(model_path, map_location=device)

local_new_save_path = '/Users/gallevshalev/Desktop/trained_models/{}/NEW_{}'.format(args.model, model_name)
server_new_save_path = '/yoav_stg/gshalev/image_captioning/{}/NEW_{}'.format(args.model, model_name)
save_path = local_new_save_path if args.run_local else server_new_save_path
torch.save({
    'epoch':saved_dict['epoch'],
    'epochs_since_improvement':saved_dict['epochs_since_improvement'],
    'bleu-4': saved_dict['bleu-4'],
    'representations':saved_dict['representations'],
    'encoder':saved_dict['encoder'].state_dict(),
    'decoder':saved_dict['decoder'].state_dict()
}, save_path)

print('saved to :{}'.format(save_path))

# load_and_resave_models.py