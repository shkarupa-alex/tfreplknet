#!/usr/bin/env python3
import argparse
import gdown
import os
import tfreplknet
import torch

CHECKPOINTS = {
    'rep_l_k_net_31_b_224_k1': 'https://drive.google.com/file/d/1DslZ2voXZQR1QoFY9KnbsHAeF84hzS0s/view?usp=sharing',
    'rep_l_k_net_31_b_224_k21': 'https://drive.google.com/file/d/1PYJiMszZYNrkZOeYwjccvxX8UHMALB7z/view?usp=sharing',
    'rep_l_k_net_31_b_384_k1': 'https://drive.google.com/file/d/1Sc46BWdXXm2fVP-K_hKKU_W8vAB-0duX/view?usp=sharing',
    # 'rep_l_k_net_31_b_384_k21': '',
    'rep_l_k_net_31_l_384_k1': 'https://drive.google.com/file/d/1JYXoNHuRvC33QV1pmpzMTKEni1hpWfBl/view?usp=sharing',
    'rep_l_k_net_31_l_384_k21': 'https://drive.google.com/file/d/16jcPsPwo5rko7ojWS9k_W-svHX-iFknY/view?usp=sharing',
    'rep_l_k_net_27_xl_320_k1': 'https://drive.google.com/file/d/1tPC60El34GntXByIRHb-z-Apm4Y5LX1T/view?usp=sharing',
    'rep_l_k_net_27_xl_320_m73': 'https://drive.google.com/file/d/1CBHAEUlCzoHfiAQmMIjZhDMAIyHUmAAj/view?usp=sharing',
}
TF_MODELS = {
    'rep_l_k_net_31_b_224_k1': tfreplknet.RepLKNetB224In1k,
    'rep_l_k_net_31_b_224_k21': tfreplknet.RepLKNetB224In21k,
    'rep_l_k_net_31_b_384_k1': tfreplknet.RepLKNetB384In1k,
    # 'rep_l_k_net_31_b_384_k21': '',
    'rep_l_k_net_31_l_384_k1': tfreplknet.RepLKNetL384In1k,
    'rep_l_k_net_31_l_384_k21': tfreplknet.RepLKNetL384In1k,
    'rep_l_k_net_27_xl_320_k1': tfreplknet.RepLKNetXL320In1k,
    'rep_l_k_net_27_xl_320_k21': tfreplknet.RepLKNetXL320In21k
}


def convert_name(name):
    name = name.replace(':0', '').replace('/', '.')
    name = name.replace('depthwise_kernel', 'weight').replace('kernel', 'weight')
    name = name.replace('large_weight', 'large_kernel')  # fix
    name = name.replace('moving_mean', 'running_mean').replace('moving_variance', 'running_var')
    name = name.replace('gamma', 'weight').replace('beta', 'bias')

    return name


def convert_weight(weight, name):
    if '/depthwise_kernel' in name and 4 == len(weight.shape):
        return weight.transpose([2, 3, 0, 1])

    if '/kernel' in name and 4 == len(weight.shape):
        return weight.transpose([2, 3, 1, 0])

    if '/kernel' in name and 2 == len(weight.shape):
        return weight.T

    return weight


if '__main__' == __name__:
    parser = argparse.ArgumentParser(
        description='Re-parameterized Large Kernel Network weight conversion from PyTorch to TensorFlow')
    parser.add_argument(
        'model_type',
        type=str,
        choices=list(CHECKPOINTS.keys()),
        help='Model checkpoint to load')
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save TensorFlow model weights')

    argv, _ = parser.parse_known_args()
    assert os.path.exists(argv.out_path) and os.path.isdir(argv.out_path), 'Wrong output path'

    weights_path = os.path.join(argv.out_path, f'{argv.model_type}.pth')
    gdown.download(url=CHECKPOINTS[argv.model_type], output=weights_path, quiet=False, fuzzy=True, resume=True)
    weights_torch = torch.load(weights_path, map_location=torch.device('cpu'))

    model = TF_MODELS[argv.model_type](weights=None)

    weights_tf = []
    for w in model.weights:

        name = convert_name(w.name)
        if name.startswith('head.') and \
                ('head.cls_22k.weight' in weights_torch or 'head.cls_22k.bias' in weights_torch):
            name = name.replace('head.', 'head.cls_22k.')
        assert name in weights_torch, f'Can\'t find weight {name} in checkpoint'

        weight = weights_torch.pop(name).numpy()
        weight = convert_weight(weight, w.name)

        weights_tf.append(weight)

    model.set_weights(weights_tf)
    model.save_weights(weights_path.replace('.pth', '.h5'), save_format='h5')
