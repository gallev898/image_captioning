import glob
from dominate import document
from dominate.tags import *
import os


model = 'run_8'

test_prefix = 'pos_dic_test_'
custom_prefix = 'pos_dic_custom_'
random_prefix = 'pos_dic_random_'

decoding_strategies = ['beam_1', 'beam_5', 'beam_10', 'top_k_5', 'top_k_10', 'top_p_0.8', 'top_p_0.9']

for ds in decoding_strategies:

    custom_model_name = '{}{}'.format(custom_prefix, ds)
    test_model_name = '{}{}'.format(test_prefix, ds)
    random_model_name = '{}{}'.format(random_prefix, ds)


    dic_test_path = os.path.join(model, '{}'.format(test_model_name))

    pos_dic_test = glob.glob(os.path.join(model, '{}/*.png'.format(test_model_name)))
    pos_dic_custom = glob.glob(os.path.join(model, '{}/*.png'.format(custom_model_name)))
    pos_dic_random = glob.glob(os.path.join(model, '{}/*.png'.format(random_model_name)))



    with document(title='Photos') as doc:
        # h1('Photos')
        with table().add(tbody()):
            l = tr()
            l += td(custom_model_name)
            l += td(random_model_name)
            with l:
                l.add(td(test_model_name))

            dic = dict()
            for path1 in pos_dic_custom:
                png = path1[path1.rindex('/') + 1:]
                x_ = [x for x in pos_dic_test if png in x]
                x_2 = [x for x in pos_dic_random if png in x]
                if len(x_) > 0 and len(x_2) >0:
                    dic[path1] = (x_[0], x_2[0])

            for path1, path2 in dic.items():
                l2 = tr()
                l2 += td(img(src=path1), _class='photo')
                l2 += td(img(src=path2[1]), _class='photo')
                with l:
                    l2.add(td(img(src=path2[0]), _class='photo'))

    # if not os.path.exists():


    with open('gallery.html', 'w') as f:
        f.write(doc.render())
