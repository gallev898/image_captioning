import glob
from dominate import document
from dominate.tags import *
import os


model = 'run_8'

test_prefix = 'pos_dic_test_'
custom_prefix = 'pos_dic_custom_'
random_prefix = 'pos_dic_random_'
cartoon_prefix = 'pos_dic_cartoon_'
cropped_images_prefix = 'pos_dic_cropped_images_'
jpeg_images_prefix = 'pos_dic_perturbed_jpeg_'
salt_images_prefix = 'pos_dic_perturbed_salt_'
aug_cartoon_images_prefix = 'NEW_pos_dic_aug_cartoon_'
snow_images_prefix = 'NEW_pos_dic_snow_'

decoding_strategies = ['beam_10']
# decoding_strategies = ['beam_1', 'beam_5', 'beam_10', 'top_k_5', 'top_k_10', 'top_p_0.8', 'top_p_0.9']

for ds in decoding_strategies:

    custom_model_name = '{}{}'.format(custom_prefix, ds)
    test_model_name = '{}{}'.format(test_prefix, ds)
    random_model_name = '{}{}'.format(random_prefix, ds)
    cartoon_model_name = '{}{}'.format(cartoon_prefix, ds)
    cropped_images_name = '{}{}'.format(cropped_images_prefix, ds)
    jpeg_images_name = '{}{}'.format(jpeg_images_prefix, ds)
    salt_images_name = '{}{}'.format(salt_images_prefix, ds)
    aug_cartoon_images_name = '{}{}'.format(aug_cartoon_images_prefix, ds)
    snow_images_name = '{}{}'.format(snow_images_prefix, ds)

    dic_test_path = os.path.join(model, '{}'.format(test_model_name))

    pos_dic_test = glob.glob(os.path.join(model, '{}/*.png'.format(test_model_name)))
    pos_dic_custom = glob.glob(os.path.join(model, '{}/*.png'.format(custom_model_name)))
    pos_dic_random = glob.glob(os.path.join(model, '{}/*.png'.format(random_model_name)))
    pos_dic_cartoon = glob.glob(os.path.join(model, '{}/*.png'.format(cartoon_model_name)))
    pos_dic_cropped_images = glob.glob(os.path.join(model, '{}/*.png'.format(cropped_images_name)))
    pos_dic_jpeg_images = glob.glob(os.path.join(model, '{}/*.png'.format(jpeg_images_name)))
    pos_dic_salt_images = glob.glob(os.path.join(model, '{}/*.png'.format(salt_images_name)))
    pos_dic_snow_images = glob.glob(os.path.join(model, '{}/*.png'.format(snow_images_name)))
    pos_dic_aug_cartoon_images = glob.glob(os.path.join(model, '{}/*.png'.format(aug_cartoon_images_name)))

    with document(title='Photos') as doc:
        # h1('Photos')
        with table().add(tbody()):
            l = tr()
            l += td(custom_model_name)
            l += td(random_model_name)
            l += td(test_model_name)
            l += td(cartoon_model_name)
            l += td(cropped_images_name)
            l += td(jpeg_images_name)
            l += td(aug_cartoon_images_name)
            l += td(snow_images_name)
            with l:
                l.add(td(salt_images_name))

            dic = dict()
            for custom in pos_dic_custom:
                png = custom[custom.rindex('/') + 1:]
                random = [x for x in pos_dic_random if png in x]
                test = [x for x in pos_dic_test if png in x]
                cartoon = [x for x in pos_dic_cartoon if png in x]
                cropped_images = [x for x in pos_dic_cropped_images if png in x]
                jpeg_images = [x for x in pos_dic_jpeg_images if png in x]
                aug_cartoon_images = [x for x in pos_dic_aug_cartoon_images if png in x]
                snow_images = [x for x in pos_dic_snow_images if png in x]
                salt_images = [x for x in pos_dic_salt_images if png in x]
                # if len(test) > 0 and len(random) >0:
                dic[custom] = (random[0] if len(random) > 0 else None,
                               test[0] if len(test)>0 else None,
                               cartoon[0] if len(cartoon)>0 else None,
                               cropped_images[0] if len(cropped_images)>0 else None,
                               jpeg_images[0] if len(jpeg_images)>0 else None,
                               aug_cartoon_images[0] if len(aug_cartoon_images)>0 else None,
                               snow_images[0] if len(snow_images)>0 else None,
                               salt_images[0] if len(salt_images)>0 else None)
                    # dic[custom] = (random[0], test[0], cartoon[0], cropped_images[0])

            for cus, others in dic.items():
                l2 = tr()
                l2 += td(img(src=cus), _class='photo', label='custom')
                l2 += td(img(src=others[0]), _class='photo', label='random') if others[0] != None else td()
                l2 += td(img(src=others[1]), _class='photo', label='test') if others[1] != None else td()
                l2 += td(img(src=others[2]), _class='photo', label='cartoon')if others[2] != None else td()
                l2 += td(img(src=others[3]), _class='photo', label='crppped')if others[3] != None else td()
                l2 += td(img(src=others[4]), _class='photo', label='jpeg')if others[4] != None else td()
                l2 += td(img(src=others[5]), _class='photo', label='aug_cartoon')if others[5] != None else td()
                l2 += td(img(src=others[6]), _class='photo', label='snow')if others[6] != None else td()
                with l:
                    l2.add(td(img(src=others[7]), _class='photo', label='salt')) if others[7] != None else l2.add(td())

    # if not os.path.exists():


    with open('{}-gallery.html'.format(ds), 'w') as f:
        f.write(doc.render())
