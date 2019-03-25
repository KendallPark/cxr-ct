from __future__ import absolute_import, division, print_function
from scipy import misc
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
import numpy as np
import imageio as imio
import matplotlib.pyplot as plt


def hg_model():
    input = keras.layers.Input((256, 256, 1))

    # Layer 1.1
    bn_conv1_b = basic_block(input, 7, 64, 2, 'Y', 'conv1_b', 'bn_conv1_b')

    # Layer 1.2
    bn1_branch1 = basic_block(bn_conv1_b, 1, 128, 1, 'N', 'res1_branch1', 'bn1_branch1')

    # Layer 1.5
    output_res = res_block(bn1_branch1, 64, 64, 128, 'N', 'N', '1', 'Nil', 'Nil')
    bn1_branch2c = output_res['output_back']

    # Layer 4
    res1 = keras.layers.Add(name='res1')([bn1_branch1, bn1_branch2c])

    output_res = res_block(res1, 64, 64, 128, 'Y', 'N', '4', 'res1_relu', 'Nil')
    res1_relu = output_res['output_front_relu']
    bn4_branch2c = output_res['output_back']

    # Layer 5
    res4 = keras.layers.Add(name='res4')([res1_relu, bn4_branch2c])
    res4_relu = keras.layers.Activation('relu', name='res4_relu')(res4)

    output_res = res_block(res4, 64, 64, 128, 'Y', 'N', '5', 'res4_relu', 'Nil')
    res4_relu = output_res['output_front_relu']
    bn5_branch2c = output_res['output_back']

    # Layer 6
    res5 = keras.layers.Add(name='res5')([res4_relu, bn5_branch2c])

    res5_relu = keras.layers.Activation('relu', name='res5_relu')(res5)
    bn6_branch1 = basic_block(res5_relu, 1, 256, 1, 'N', 'res6_branch1', 'bn6_branch1')

    output_res = res_block(bn6_branch1, 128, 128, 256, 'N', 'N', '6', 'Nil', 'Nil')
    bn6_branch2c = output_res['output_back']

    # layer L6^0
    res6 = keras.layers.Add(name='res6')([bn6_branch1, bn6_branch2c])
    res6_0_layer_names = {'relu_1': 'res6_relu', 'max_pool_1': 'hg1_pool1', 'add_1': 'reshg1_low1' \
        , 'relu_2': 'reshg1_low1_relu', 'max_pool_2': 'Nil', 'add_2': 'reshg1_low2' \
        , 'relu_5': 'reshg1_low2_relu', 'max_pool_5': 'Nil'}
    output_down = down_block(res6, 'hg1', res6_0_layer_names)

    res6_relu = output_down['output_front1']
    reshg1_low2_relu = output_down['output_front5']
    bnhg1_low5_branch2c = output_down['output_back']

    # layer L6^1
    reshg1_low5 = keras.layers.Add(name='reshg1_low5')([reshg1_low2_relu, bnhg1_low5_branch2c])

    res6_1_layer_names = {'relu_1': 'reshg1_low5_relu', 'max_pool_1': 'hg1_low6_pool1', 'add_1': 'reshg1_low6_low1' \
        , 'relu_2': 'reshg1_low6_low1_relu', 'max_pool_2': 'Nil', 'add_2': 'reshg1_low6_low2' \
        , 'relu_5': 'reshg1_low6_low2_relu', 'max_pool_5': 'Nil'}
    output_down = down_block(reshg1_low5, 'hg1_low6', res6_1_layer_names)

    reshg1_low5_relu = output_down['output_front1']
    reshg1_low6_low2_relu = output_down['output_front5']
    bnhg1_low6_low5_branch2c = output_down['output_back']

    # layer L6^2
    reshg1_low6_low5 = keras.layers.Add(name='reshg1_low6_low5')([reshg1_low6_low2_relu, bnhg1_low6_low5_branch2c])

    res6_2_layer_names = {'relu_1': 'reshg1_low6_low5_relu', 'max_pool_1': 'hg1_low6_low6_pool1',
                          'add_1': 'reshg1_low6_low6_low1' \
        , 'relu_2': 'reshg1_low6_low6_low1_relu', 'max_pool_2': 'Nil', 'add_2': 'reshg1_low6_low6_low2' \
        , 'relu_5': 'reshg1_low6_low6_low2_relu', 'max_pool_5': 'Nil'}
    output_down = down_block(reshg1_low6_low5, 'hg1_low6_low6', res6_2_layer_names)

    reshg1_low6_low5_relu = output_down['output_front1']
    reshg1_low6_low6_low2_relu = output_down['output_front5']
    bnhg1_low6_low6_low5_branch2c = output_down['output_back']

    # layer L6^3
    reshg1_low6_low6_low5 = keras.layers.Add(name='reshg1_low6_low6_low5')(
        [reshg1_low6_low6_low2_relu, bnhg1_low6_low6_low5_branch2c])

    res6_3_layer_names = {'relu_1': 'reshg1_low6_low6_low5_relu', 'max_pool_1': 'hg1_low6_low6_low6_pool1',
                          'add_1': 'reshg1_low6_low6_low6_low1' \
        , 'relu_2': 'reshg1_low6_low6_low6_low1_relu', 'max_pool_2': 'Nil', 'add_2': 'reshg1_low6_low6_low6_low2' \
        , 'relu_5': 'reshg1_low6_low6_low6_low2_relu', 'max_pool_5': 'Nil'}
    output_down = down_block(reshg1_low6_low6_low5, 'hg1_low6_low6_low6', res6_3_layer_names)

    reshg1_low6_low6_low5_relu = output_down['output_front1']
    reshg1_low6_low6_low6_low2_relu = output_down['output_front5']
    bnhg1_low6_low6_low6_low5_branch2c = output_down['output_back']

    # layer L6^3_L67
    reshg1_low6_low6_low6_low5 = keras.layers.Add(name='reshg1_low6_low6_low6_low5')(
        [reshg1_low6_low6_low6_low2_relu, bnhg1_low6_low6_low6_low5_branch2c])

    output_res = res_block(reshg1_low6_low6_low6_low5, 128, 128, 256, 'Y', 'N', 'hg1_low6_low6_low6_low6',
                           'reshg1_low6_low6_low6_low5_relu', 'Nil')
    reshg1_low6_low6_low6_low5_relu = output_res['output_front_relu']
    bnhg1_low6_low6_low6_low6_branch2c = bn6_branch2c = output_res['output_back']

    reshg1_low6_low6_low6_low6 = keras.layers.Add(name='reshg1_low6_low6_low6_low6')(
        [reshg1_low6_low6_low6_low5_relu, bnhg1_low6_low6_low6_low6_branch2c])
    output_res = res_block(reshg1_low6_low6_low6_low6, 128, 128, 256, 'Y', 'N', 'hg1_low6_low6_low6_low7',
                           'reshg1_low6_low6_low6_low6_relu', 'Nil')
    reshg1_low6_low6_low6_low6_relu = output_res['output_front_relu']
    bnhg1_low6_low6_low6_low7_branch2c = bn6_branch2c = output_res['output_back']

    ### Decoder start from here ###

    # layer DL6^2
    reshg1_low6_low6_low6_low7 = keras.layers.Add(name='reshg1_low6_low6_low6_low7')(
        [reshg1_low6_low6_low6_low6_relu, bnhg1_low6_low6_low6_low7_branch2c])
    reshg1_low6_low6_low6_low7_relu = keras.layers.Activation('relu', name='reshg1_low6_low6_low6_low7_relu')(
        reshg1_low6_low6_low6_low7)
    hg1_low6_low6_low6_up5 = keras.layers.Conv2DTranspose(kernel_size=4, filters=256, strides=2, padding='same',
                                                          name='hg1_low6_low6_low6_up5')(
        reshg1_low6_low6_low6_low7_relu)

    hg1_low6_low6_low6 = keras.layers.Add(name='hg1_low6_low6_low6')(
        [reshg1_low6_low6_low5_relu, hg1_low6_low6_low6_up5])

    output_res = res_block(hg1_low6_low6_low6, 128, 128, 256, 'N', 'N', 'hg1_low6_low6_low7', 'Nil', 'Nil')
    bnhg1_low6_low6_low7_branch2c = output_res['output_back']

    # layer DL6^1
    reshg1_low6_low6_low7 = keras.layers.Add(name='reshg1_low6_low6_low7')(
        [hg1_low6_low6_low6, bnhg1_low6_low6_low7_branch2c])

    reshg1_low6_low6_low7_relu = keras.layers.Activation('relu', name='reshg1_low6_low6_low7_relu')(
        reshg1_low6_low6_low7)
    hg1_low6_low6_up5 = keras.layers.Conv2DTranspose(kernel_size=4, filters=256, strides=2, padding='same',
                                                     name='hg1_low6_low6_up5')(reshg1_low6_low6_low7_relu)

    hg1_low6_low6 = keras.layers.Add(name='hg1_low6_low6')([reshg1_low6_low5_relu, hg1_low6_low6_up5])

    output_res = res_block(hg1_low6_low6, 128, 128, 256, 'N', 'N', 'hg1_low6_low7', 'Nil', 'Nil')
    bnhg1_low6_low7_branch2c = output_res['output_back']

    # layer DL6^0
    reshg1_low6_low7 = keras.layers.Add(name='reshg1_low6_low7')([hg1_low6_low6, bnhg1_low6_low7_branch2c])
    reshg1_low6_low7_relu = keras.layers.Activation('relu', name='reshg1_low6_low7_relu')(reshg1_low6_low7)
    hg1_low6_up5 = keras.layers.Conv2DTranspose(kernel_size=4, filters=256, strides=2, padding='same',
                                                name='hg1_low6_up5')(reshg1_low6_low7_relu)

    hg1_low6 = keras.layers.Add(name='hg1_low6')([reshg1_low5_relu, hg1_low6_up5])

    output_res = res_block(hg1_low6, 128, 128, 256, 'N', 'N', 'hg1_low7', 'Nil', 'Nil')
    bnhg1_low7_branch2c = output_res['output_back']

    # layer 1/3
    reshg1_low7 = keras.layers.Add(name='reshg1_low7')([hg1_low6, bnhg1_low7_branch2c])
    reshg1_low7_relu = keras.layers.Activation('relu', name='reshg1_low7_relu')(reshg1_low7)
    hg1_up5 = keras.layers.Conv2DTranspose(kernel_size=4, filters=256, strides=2, padding='same', name='hg1_up5')(
        reshg1_low7_relu)

    hg1 = keras.layers.Add(name='hg1')([res6_relu, hg1_up5])
    bn_linear1 = basic_block(hg1, 1, 256, 1, 'Y', 'linear1', 'bn_linear1')

    pre_output = keras.layers.Conv2D(kernel_size=1, filters=128, strides=1, padding='same', use_bias=False,
                                     name='pre_output')(bn_linear1)
#     output = keras.layers.Flatten()(pre_output)

    # this line will be removed when the loss function is constructed
#     output = keras.layers.Dense(3, activation=tf.nn.sigmoid)(output)

    model = keras.models.Model(input, pre_output)

    return model


def basic_block(input_basic, size_filter, num_filter, stride, relu, conv_name, bn_name):
    # relu can be Y or N
    conv = keras.layers.Conv2D(kernel_size=size_filter, filters=num_filter, strides=stride, padding='same',
                               use_bias=False, name=conv_name)(input_basic)
    bn = keras.layers.BatchNormalization(name=bn_name)(conv)

    if relu == 'Y':
        relu = keras.layers.Activation('relu')(bn)
        output_basic = relu
    elif relu == 'N':
        output_basic = bn
    else:
        print('error: basic_block')

    return output_basic


def res_block(input_res, num_filter1, num_filter2, num_filter3, relu_front, max_pool_front, block_name, relu_name,
              max_pool_name):
    # filter sizes and stride will be (1,1), (3,3) and (1,1) for the 3 conv().
    # there will be 2 ReLU between the 3 conv(). no ReLU at last.
    # relu_front and max_pool_front can be Y or N
    # there are optional relu() and/or max_pool() at the front
    # return_option=0 means return relu; return_option=1 means return max_pool

    conv_a_name = 'res' + block_name + '_branch2a'
    bn_a_name = 'bn' + block_name + '_branch2a'
    conv_b_name = 'res' + block_name + '_branch2b'
    bn_b_name = 'bn' + block_name + '_branch2b'
    conv_c_name = 'res' + block_name + '_branch2c'
    bn_c_name = 'bn' + block_name + '_branch2c'

    # optional relu
    if relu_front == 'Y':
        output_relu = keras.layers.Activation('relu', name=relu_name)(input_res)
    elif relu_front == 'N':
        output_relu = input_res
    else:
        print('error: res_block')

    # optional max_pool
    if max_pool_front == 'Y':
        output_max_pool = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid',
                                                    name=max_pool_name)(output_relu)
    elif max_pool_front == 'N':
        output_max_pool = output_relu
    else:
        print('error: res_block')

    output_basic1 = basic_block(output_max_pool, 1, num_filter1, 1, 'Y', conv_a_name, bn_a_name)
    output_basic2 = basic_block(output_basic1, 3, num_filter2, 1, 'Y', conv_b_name, bn_b_name)
    output_basic3 = basic_block(output_basic2, 1, num_filter3, 1, 'N', conv_c_name, bn_c_name)

    output_res = {'output_front_relu': output_relu, 'output_front_max_pool': output_max_pool,
                  'output_back': output_basic3}

    return output_res


def down_block(input_down, block_name, layer_names):
    # for number of filters, should use (128,128,256)x3
    # when testing, use (2,2,2)x3
    num_filter_a = 128
    num_filter_b = 128
    num_filter_c = 256

    res_1_name = block_name + '_low1'
    res_2_name = block_name + '_low2'
    res_5_name = block_name + '_low5'

    relu_1_name = layer_names['relu_1']
    max_pool_1_name = layer_names['max_pool_1']
    add_1_name = layer_names['add_1']

    relu_2_name = layer_names['relu_2']
    max_pool_2_name = layer_names['max_pool_2']
    add_2_name = layer_names['add_2']

    relu_5_name = layer_names['relu_5']
    max_pool_5_name = layer_names['max_pool_5']

    output_res1 = res_block(input_down, num_filter_a, num_filter_b, num_filter_c, 'Y', 'Y', res_1_name, relu_1_name,
                            max_pool_1_name)
    output_res1_front_relu = output_res1['output_front_relu']
    output_res1_front_max_pool = output_res1['output_front_max_pool']
    output_res1_back = output_res1['output_back']
    output_res1_add = keras.layers.Add(name=add_1_name)([output_res1_front_max_pool, output_res1_back])

    output_res2 = res_block(output_res1_add, num_filter_a, num_filter_b, num_filter_c, 'Y', 'N', res_2_name,
                            relu_2_name, max_pool_2_name)
    output_res2_front = output_res2['output_front_relu']
    output_res2_back = output_res2['output_back']
    output_res2_add = keras.layers.Add(name=add_2_name)([output_res2_front, output_res2_back])

    output_res5 = res_block(output_res2_add, num_filter_a, num_filter_b, num_filter_c, 'Y', 'N', res_5_name,
                            relu_5_name, max_pool_5_name)
    output_res5_front = output_res5['output_front_relu']
    output_res5_back = output_res5['output_back']

    output_down = {'output_front1': output_res1_front_relu, 'output_front5': output_res5_front,
                   'output_back': output_res5_back}

    return output_down


def add_weights(model):
    # add the pre-existing weights to the model
    np_weights = np.load('C:/Users/Lanston/Desktop/US Life/MLSP-Courses-UW-Madison/CS 766 - Computer Vision/Project/Model/tf_conversion/skulls_tf.npy', encoding='latin1')
    layers = np_weights[()].keys()
    key_weight=np_weights[()]['conv1_b'].keys()
    
    for layer in model.layers:
        if layer.name in layers:
            # only upload for layers with weight
            if np_weights[()][layer.name].keys() == key_weight:
                #print(layer.name)
                weight = np_weights[()][layer.name]['weights']
                # fix discrepancies on 6 layers
                if layer.name in ['hg1_low6_low6_low6_up5', 'hg1_low6_low6_up5', 'hg1_low6_up5', 'hg1_up5']:
                  weight = np.repeat(weight, 256, axis=2)
                  layer.set_weights([weight, np.zeros(256)])
                elif layer.name in ['res1_branch2a', 'res6_branch2a']:
                  weight = np.repeat(weight, 2, axis=2)
                  layer.set_weights([weight])
                else:
                  layer.set_weights([weight])
                  
               

def print_weight(model):
    num_layers=len(model.layers)

    for i in range(num_layers):
        print('############################################')
        print(model.layers[i].get_config()['name'])
        try:
            print(model.layers[i].get_weights()[0].shape)
        except:
            print('no weight for this layer')
    return


model = hg_model()

model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['accuracy', 'mean_squared_error'])

# print(model.summary())

add_weights(model)

# print_weight(model)

filepath='C:/Users/Lanston/Desktop/data/'
filename1=filepath+'img001Grey.png'
filename2=filepath+'img002Grey.png'

f1 = imio.imread(filename1)
f2 = imio.imread(filename2)
fT1 = np.reshape(f1, (256, 256, 1))/256
fT2 = np.reshape(f2, (256, 256, 1))/256

fT1 = np.expand_dims(fT1, axis = 0)
out = model.predict(fT1)

out = out[0]

print(out[:,:,1])

# train_images = np.stack((fT1, fT2))
# train_labels = np.array([[2, 8, 9], [1, 0, 7]])


# model.fit(train_images, train_labels, epochs=2)