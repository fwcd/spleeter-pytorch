import math
import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F

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

def unfold(x: torch.Tensor, size: int, step: int) -> torch.Tensor:
    '''
    Extract sliding windows from a 1D tensor purely in terms
    of `torch.nn.functional.unfold`. This serves as a polyfill
    for `torch.unfold` which coremltools does not support yet.

    `unfold(x, size, step) == x.unfold(0, size, step)`
    '''

    assert len(x.shape) == 1

    # Reshape to 4D
    x = x[None, None, None]

    # Apply unfold
    y = F.unfold(x, kernel_size=(1, size), stride=step)
    y = y.transpose(-1, -2)

    # Remove unneeded dimensions again
    y = y[0]

    return y

# Source: https://github.com/kaituoxu/Conv-TasNet/blob/master/src/utils.py
# MIT-licensed, Copyright (c) 2018 Kaituo XU

def overlap_and_add(signal: torch.Tensor, frame_step: int):
    '''
    Reconstructs a signal from a framed representation.
    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where
        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    '''
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).float() # floats are needed to unfold with torch.nn.functional
    frame = unfold(frame, subframes_per_frame, subframe_step)
    frame = frame.clone().long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result
