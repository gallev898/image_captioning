import glob
from dominate import document
from dominate.tags import *
import os


test_prefix = 'pos_dic_test_'
decoding_strategies = ['beam_1', 'beam_5', 'beam_10', 'top_k_5', 'top_k_10', 'top_p_0.8', 'top_p_0.9']
models = ['unlikelihood_full_replace_16_16', 'unlikelihood_noun_replace_16_16', 'run_batch_size_16']
png_to_show = ['VERB logits.png', 'ADV logits.png', 'PRON exp.png', 'NOUN logits.png', 'alphas max avg.png', 'count.png']
data = []
# filterd = False
filterd = True

for model in models:
    # section: for every decoding strategy
    for ds in decoding_strategies:
        test_model_name = '{}{}'.format(test_prefix, ds)
        dic_test_path = os.path.join(model, '{}'.format(test_model_name))
        d = glob.glob(os.path.join(model, '{}/*.png'.format(test_model_name)))
        if len(d) > 0:
            data.append(d)

document = document(title='Photos')
table = table()
tbody= tbody()

l = tr()
for d in data:
    l += td(d[0][0:d[0].index('/')])
table.add(l)


def build():
    for i in range(1, len(data)):
        cur_lst = data[i]
        t = [pp for pp in cur_lst if png in pp]
        if len(t) > 0:
            cur_png = t[0]
            cur_pngs.append(cur_png)
    l2 = tr()
    l2 += td(img(src=png_path), _class='photo', label='')
    for p in range(0, len(cur_pngs)):
        l2 += td(img(src=cur_pngs[p]), _class='photo', label='')
    tbody.add(l2)


for png_path in data[0]:
    cur_pngs = []
    png = png_path[png_path.rindex('/') + 1:]
    if filterd:
        if png in png_to_show:
            build()
    else:
        build()
        #######

        #######


table.add(tbody)
document.add(table)
with open('{}-gallery.html'.format(''), 'w') as f:
    f.write(document.render())
#