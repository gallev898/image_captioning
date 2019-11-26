import json


word = str('Otter').lower()

with open('word_to_count.json') as json_file:
    word_to_count = json.load(json_file)

with open('open_images_classes.json') as json_file:
    open_images_classes = json.load(json_file)

if word in word_to_count:
    print('{} exist {} times'.format(word, word_to_count[word]))
else:
    print('{} DONT exist'.format(word))
