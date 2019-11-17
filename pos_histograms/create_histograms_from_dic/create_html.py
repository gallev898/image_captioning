import glob
from dominate import document
from dominate.tags import *
import os


model = 'standart_training_with_fine_tune_after_13_epochs_run6'
pos_dic_perturbed_test_fog_beam_5 = glob.glob(os.path.join(model, 'pos_dic_custom_beam_5/*.png'))
pos_dic_test_beam_5 = glob.glob(os.path.join(model, 'pos_dic_test_beam_5/*.png'))

with document(title='Photos') as doc:
    # h1('Photos')
    with table().add(tbody()):
        l = tr()
        l += td('pos_dic_perturbed_test_fog_beam_5')
        with l:
            l.add(td('pos_dic_test_beam_5'))

        dic = dict()
        for path1 in pos_dic_perturbed_test_fog_beam_5:
            png = path1[path1.rindex('/') + 1:]
            x_ = [x for x in pos_dic_test_beam_5 if png in x]
            if len(x_) > 0:
                dic[path1] = x_[0]

        for path1, path2 in dic.items():
            l2 = tr()
            l2 += td(img(src=path1), _class='photo')
            with l:
                l2.add(td(img(src=path2), _class='photo'))

with open('gallery.html', 'w') as f:
    f.write(doc.render())
