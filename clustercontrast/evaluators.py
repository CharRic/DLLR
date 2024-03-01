from __future__ import print_function, absolute_import

import time
from collections import OrderedDict

import torch

from .utils import to_torch
from .utils.meters import AverageMeter


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def extract_cnn_feature(model, inputs, mode):
    inputs = to_torch(inputs).cuda()
    # inputs1 = inputs
    # print(inputs)
    outputs = model(inputs, inputs, modal=mode)
    outputs = outputs.data.cpu()
    return outputs


def extract_features(model, data_loader, print_freq=50, mode=0):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, mode)
            flip = fliplr(imgs)
            # print(flip)
            outputs_flip = extract_cnn_feature(model, flip, mode)

            for fname, output, output_flip, pid in zip(fnames, outputs, outputs_flip, pids):
                features[fname] = (output.detach() + output_flip.detach()) / 2.0
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'.format(i + 1, len(data_loader), batch_time.val, batch_time.avg,
                                                      data_time.val, data_time.avg))

    return features, labels
