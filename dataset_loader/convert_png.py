import cv2
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--run_local', action='store_true', default=False)
args = parser.parse_args()

if args.run_local:
    cartoon_dir = '/home/gal/Desktop/cartoons'
else:
    cartoon_dir = '/yoav_stg/gshalev/semantic_labeling/cartoon/cartoonset10k'
files = os.listdir(cartoon_dir)

ctr = 0
for f in files:
    ctr += 1
    if ctr % 100 == 0:
        print('{}/{}'.format(ctr, len(files)))
    image = cv2.imread('{}/{}'.format(cartoon_dir,f), cv2.IMREAD_UNCHANGED)

    #make mask of where the transparent bits are
    trans_mask = image[:,:,3] == 0

    #replace areas of transparency with white and not transparent
    image[trans_mask] = [255, 255, 255, 255]

    #new image without alpha channel...
    new_img = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    cv2.imwrite('{}/{}'.format(cartoon_dir,f), new_img)
print('done')