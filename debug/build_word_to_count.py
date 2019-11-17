from dataset_loader.datasets import CaptionDataset
from standart_training.utils import *
from utils import *

data_folder = '/yoav_stg/gshalev/image_captioning/output_folder'  # folder with data files saved by create_input_files.py
# data_folder = 'output_folder'  # folder with data files saved by create_input_files.py

train_loader = torch.utils.data.DataLoader(
    CaptionDataset(data_folder, data_name, 'TRAIN', transform=transforms.Compose([data_normalization])),
    batch_size=1, shuffle=True, num_workers=1, pin_memory=True)

word_map_file = os.path.join(data_folder, 'WORDMAP_' + data_name + '.json')

with open(word_map_file, 'r') as j:
    word_map = json.load(j)

rev_word_map = {v: k for k, v in word_map.items()}

word_dic = dict()
for i, (imgs, caps, caplens) in tqdm(enumerate(train_loader)):
    if i % 1000 == 0:
        print('{}/{}'.format(i, len(train_loader)))
    numpy_ = [rev_word_map[x] for x in caps[0].numpy()][1:caplens - 1]
    for n in numpy_:
        word_dic[n] = word_dic[n] + 1 if n in word_dic else 1

with open('word_to_count.json', 'w') as fp:
    json.dump(word_dic, fp)

# debug.py
