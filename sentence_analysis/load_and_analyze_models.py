import torch
import numpy as np
import argparse

from torchvision.transforms import transforms

from dataset_loader.datasets import CaptionDataset


# subsec: arg
parser = argparse.ArgumentParser()
parser.add_argument('--run_local', default=False, action='store_true')
parser.add_argument('--modelA', type=str)
parser.add_argument('--modelB', type=str)
parser.add_argument('--decoding', type=str, default='beam_1')
parser.add_argument('--model_file_name', type=str,
                    default='NEW_BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar')
args = parser.parse_args()

coco_data_path = '/Users/gallevshalev/PycharmProjects/image_captioning/output_folder' if args.run_local else '/yoav_stg/gshalev/image_captioning/output_folder'
data_name = 'coco_5_cap_per_img_5_min_word_freq'
data_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print('args.modelA: {}'.format(args.modelA))
print('args.modelB: {}'.format(args.modelB))

# subsec: initialization
path = '/Users/gallevshalev/Desktop/trained_models/{}/inference_data/{}_test_{}'

model_pathA_metric = path.format(args.modelA, 'metrics_results', args.decoding)
model_pathB_metric = path.format(args.modelB, 'metrics_results', args.decoding)

model_pathA_pos_dic = path.format(args.modelA, 'pos_dic', args.decoding)
model_pathB_pos_dic = path.format(args.modelB, 'pos_dic', args.decoding)

# subsec: load model
modelA_metric = torch.load(model_pathA_metric)
modelB_metric = torch.load(model_pathB_metric)

modelA_pos_dic = torch.load(model_pathA_pos_dic)
modelB_pos_dic = torch.load(model_pathB_pos_dic)

generated_sentencesA = modelA_metric['hyp']['annotations']
generated_sentencesB = modelB_metric['hyp']['annotations']

# section: avg sentence len
avg_sentence_lenA = sum([len(x['caption']) for x in generated_sentencesA])/len(generated_sentencesA)
avg_sentence_lenB = sum([len(x['caption']) for x in generated_sentencesB])/len(generated_sentencesB)

modelA_sentence_likelihood = [x[1] for x in modelA_pos_dic['generated_sentences_likelihood']]
modelB_sentence_likelihood = [x[1] for x in modelB_pos_dic['generated_sentences_likelihood']]

# section: avg sentence likelihood
modelA_sentences_likelihood_avg = np.average(modelA_sentence_likelihood)
modelB_sentences_likelihood_avg = np.average(modelB_sentence_likelihood)


vocabulary_usage_modelA = set()
vocabulary_usage_modelB = set()

[vocabulary_usage_modelA.update(x['caption']) for x in generated_sentencesA]
[vocabulary_usage_modelB.update(x['caption']) for x in generated_sentencesB]

# section: vocabulary usage count
vocabulary_usage_modelA_count = len(vocabulary_usage_modelA)
vocabulary_usage_modelB_count = len(vocabulary_usage_modelB)

f = 0

print(' '.join(list(filter(lambda x: x['image_id'] == 315, generated_sentencesA))[0]['caption']))
print(' '.join(list(filter(lambda x: x['image_id'] == 315, generated_sentencesB))[0]['caption']))

f = CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization]))
# test_loader = torch.utils.data.DataLoader(
#     CaptionDataset(coco_data_path, data_name, 'TEST', transform=transforms.Compose([data_normalization])),
#     batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
d=0