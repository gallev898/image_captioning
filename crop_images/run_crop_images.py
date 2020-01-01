import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run_local', action='store_true', default=False)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--ex_limit', type=int, default=10)
args = parser.parse_args()

if args.run_local:
    croped_images_folder = 'cropped_images'
    origin_img_folder = 'images'
    descriptions_file_name = 'data/class-descriptions-boxable.csv'
    annotations_file_names = ['data/train/train-annotations-bbox.csv', 'data/test/test-annotations-bbox.csv']

else:
    croped_images_folder = '/yoav_stg/gshalev/semantic_labeling/cropped_open_images'
    origin_img_folder = '/yoav_stg/gshalev/semantic_labeling/open_images'
    descriptions_file_name = '/yoav_stg/gshalev/image_captioning/output_folder/class-descriptions-boxable.csv'
    annotations_file_names = ['/yoav_stg/gshalev/image_captioning/output_folder/train-annotations-bbox.csv',
                              '/yoav_stg/gshalev/image_captioning/output_folder/test-annotations-bbox.csv']

images_ids_to_label_name = {}

for img in os.listdir(origin_img_folder):
    images_ids_to_label_name[img.split('_')[0]] = img.split('_')[1].split('.')[0]

class_code_to_class_name = dict()
class_name_to_class_code = dict()

# section: Map class to code
with open(descriptions_file_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        class_code_to_class_name[row[0]] = row[1]
        class_name_to_class_code[row[1]] = row[0]

# section: Croped images
if not os.path.exists(croped_images_folder):
    os.mkdir(croped_images_folder)

ex_count = 0
for file_name in annotations_file_names:
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                print(f'Column names are {", ".join(row)}')
                line_count += 1
            else:
                # print('ex_count: {} line_counter: {}'.format(ex_count, line_count))
                if args.debug and ex_count > args.ex_limit:
                    print('break after {} examples'.format(ex_count))
                    break
                image_id = row[0]
                image_label_code = row[2]
                image_label = class_code_to_class_name[image_label_code]

                if image_id in images_ids_to_label_name.keys() and image_label == images_ids_to_label_name[image_id]:

                    # section: Get image
                    img_name = image_id + '_' + images_ids_to_label_name[image_id] + '.jpg'
                    img = Image.open(os.path.join(origin_img_folder, img_name))

                    if args.run_local:
                        # section: Show image
                        plt.imshow(img)
                        plt.show()

                    # section: Crop image
                    img2 = img.crop((float(row[4]) * img.width, float(row[6]) * img.height, float(row[5]) * img.width,
                                     float(row[7]) * img.height))

                    if args.run_local:
                        # section: Show croped image
                        plt.imshow(img2)
                        plt.show()

                    # section: Save croped image
                    print('saving: {}'.format(os.path.join(croped_images_folder, img_name)))
                    img2.save(os.path.join(croped_images_folder, img_name))

                    ex_count += 1
                line_count += 1

# run_crop_images.py
