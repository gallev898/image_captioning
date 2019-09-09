import sys
sys.path.append('/home/mlspeech/gshalev/gal/image_captioning')

from utils import create_input_files
import os
import argparse

# wandb login a8c4526db3e8aa11d7b2674d7c257c58313b45ca
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create input files')
    parser.add_argument('--run_local', default=False, action='store_true')
    parser.add_argument('--captions_per_image', default=5, type=int)
    parser.add_argument('--max_len', default=50, type=int)
    parser.add_argument('--min_word_freq', default=5, type=int)
    parser.add_argument('--output_folder', default='output_folder', type=str)
    args = parser.parse_args()

    # Create input files (along with word map)
    if args.run_local:
        desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')
        karpathy_json_path = os.path.join(desktop_path, 'datasets/mscoco/dataset_coco.json')

        image_folder = os.path.join(desktop_path, 'datasets/mscoco')
        splits_json_path = karpathy_json_path

        if not os.path.exists(args.output_folder):
            os.mkdir(args.output_folder)

    else:
        image_folder = '/yoav_stg/gshalev/semantic_labeling/mscoco'
        splits_json_path = '/yoav_stg/gshalev/semantic_labeling/mscoco/dataset_coco.json'

    if not os.path.exists(args.output_folder):
        os.mkdir(args.output_folder)

    create_input_files(limit=args.run_local, dataset='coco',
                       karpathy_json_path=splits_json_path,
                       image_folder=image_folder,
                       captions_per_image=args.captions_per_image,
                       min_word_freq=args.min_word_freq,
                       output_folder=args.output_folder,
                       max_len=args.max_len)
