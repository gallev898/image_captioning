import glob
from dominate import document
from dominate.tags import *
import os


model = 'run_7'

custom_to_test = {}
to_take_care_of = []
for pos_dic in os.listdir(model):
    print(pos_dic)
    if 'custom' in pos_dic:
        custom_to_test[pos_dic] = pos_dic.replace('custom', 'test')

graph_name = 'ADJ exp.png'
with document(title=graph_name.split('.')[0]) as doc:
    with table().add(tbody()):
        for custome, test in custom_to_test.items():
            l = tr()
            l += td(custome)
            with l:
                l.add(td(test))

            custome_path = 'run_7/{}/{}'.format(custome, graph_name)
            if not os.path.exists(custome_path):
                custome_path = custome_path.replace('exp', 'prop')
            test_path = 'run_7/{}/{}'.format(test, graph_name)
            if not os.path.exists(test_path):
                test_path = test_path.replace('exp', 'prop')
            l2 = tr()
            l2 += td(img(src=custome_path), _class='photo')
            with l:
                l2.add(td(img(src=test_path), _class='photo'))

with open('{}_{}.html'.format(model, graph_name.split('.')[0]), 'w') as f:
    f.write(doc.render())
