import numpy as np
import tensorflow as tf

from pathlib import Path

def tf2pytorch(checkpoint_path: Path, num_instruments: int):
    tf_vars = {}
    init_vars = tf.train.list_variables(checkpoint_path)
    # print(init_vars)
    for name, shape in init_vars:
        try:
            # print('Loading TF Weight {} with shape {}'.format(name, shape))
            data = tf.train.load_variable(checkpoint_path, name)
            tf_vars[name] = data
        except Exception as e:
            print('Load error')
    conv_idx = 0
    tconv_idx = 0
    bn_idx = 0
    outputs = []
    for i in range(num_instruments):
        output = {}
        outputs.append(output)

        for j in range(1,7):
            if conv_idx == 0:
                conv_suffix = ''
            else:
                conv_suffix = f'_{conv_idx}'

            if bn_idx == 0:
                bn_suffix = ''
            else:
                bn_suffix = f'_{bn_idx}'

            output[f'down{j}_conv.1.weight'] = np.transpose(tf_vars[f'conv2d{conv_suffix}/kernel'], (3, 2, 0, 1))
            # print('conv dtype: ',output[f'down{j}.0.weight'].dtype)
            output[f'down{j}_conv.1.bias'] = tf_vars[f'conv2d{conv_suffix}/bias']

            output[f'down{j}_act.0.weight'] = tf_vars[f'batch_normalization{bn_suffix}/gamma']
            output[f'down{j}_act.0.bias'] = tf_vars[f'batch_normalization{bn_suffix}/beta']
            output[f'down{j}_act.0.running_mean'] = tf_vars[f'batch_normalization{bn_suffix}/moving_mean']
            output[f'down{j}_act.0.running_var'] = tf_vars[f'batch_normalization{bn_suffix}/moving_variance']

            conv_idx += 1
            bn_idx += 1

        # up blocks
        for j in range(1, 7):
            if tconv_idx == 0:
                tconv_suffix = ''
            else:
                tconv_suffix = f'_{tconv_idx}'

            if bn_idx == 0:
                bn_suffix = ''
            else:
                bn_suffix= f'_{bn_idx}'

            output[f'up{j}.0.weight'] = np.transpose(tf_vars[f'conv2d_transpose{tconv_suffix}/kernel'], (3,2,0, 1))
            output[f'up{j}.0.bias'] = tf_vars[f'conv2d_transpose{tconv_suffix}/bias']
            output[f'up{j}.3.weight'] = tf_vars[f'batch_normalization{bn_suffix}/gamma']
            output[f'up{j}.3.bias'] = tf_vars[f'batch_normalization{bn_suffix}/beta']
            output[f'up{j}.3.running_mean'] = tf_vars[f'batch_normalization{bn_suffix}/moving_mean']
            output[f'up{j}.3.running_var'] = tf_vars[f'batch_normalization{bn_suffix}/moving_variance']
            tconv_idx += 1
            bn_idx += 1

        if conv_idx == 0:
            suffix = ''
        else:
            suffix = '_' + str(conv_idx)
        output['up7.0.weight'] = np.transpose(
            tf_vars['conv2d{}/kernel'.format(suffix)], (3, 2, 0, 1))
        output['up7.0.bias'] = tf_vars['conv2d{}/bias'.format(suffix)]
        conv_idx += 1

    return outputs
