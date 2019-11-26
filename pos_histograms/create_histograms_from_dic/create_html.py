import glob
from dominate import document
from dominate.tags import *
import os


model = 'run_7'
custom_model_name = 'pos_dic_custom_beam_10'
test_model_name = 'pos_dic_test_beam_10'
pos_dic_custom = glob.glob(os.path.join(model, '{}/*.png'.format(custom_model_name)))
pos_dic_test = glob.glob(os.path.join(model, '{}/*.png'.format(test_model_name)))

with document(title='Photos') as doc:
    # h1('Photos')
    with table().add(tbody()):
        l = tr()
        l += td(custom_model_name)
        with l:
            l.add(td(test_model_name))

        dic = dict()
        for path1 in pos_dic_custom:
            png = path1[path1.rindex('/') + 1:]
            x_ = [x for x in pos_dic_test if png in x]
            if len(x_) > 0:
                dic[path1] = x_[0]

        for path1, path2 in dic.items():
            l2 = tr()
            l2 += td(img(src=path1), _class='photo')
            with l:
                l2.add(td(img(src=path2), _class='photo'))

with open('gallery.html', 'w') as f:
    f.write(doc.render())
