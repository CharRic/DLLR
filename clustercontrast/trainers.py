from __future__ import print_function, absolute_import

import time

import torch
from torch import nn

from utils import AverageMeter


class DualModalityShard_Trainer(object):
    def __init__(self, encoder, memory_rgb=None, memory_ir=None):
        super(DualModalityShard_Trainer, self).__init__()
        self.encoder = encoder

        self.memory_rgb = memory_rgb
        self.memory_ir = memory_ir

    def train(self, epoch, data_loader_rgb=None, data_loader_ir=None, data_loader_rgb2ir=None, data_loader_ir2rgb=None,
              optimizer=None, print_freq=10, train_iters=400, stage=1):

        self.encoder.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses = AverageMeter()

        criterion_dis = nn.BCELoss()
        criterion_dis = criterion_dis.cuda()

        end = time.time()

        for i in range(train_iters):
            data_time.update(time.time() - end)

            # load data
            inputs_rgb = data_loader_rgb.next()
            inputs_ir = data_loader_ir.next()

            # process inputs
            inputs_rgb, inputs_rgb1, labels_rgb, indexes_rgb = self._parse_data_rgb(inputs_rgb)
            inputs_ir, labels_ir, indexes_ir = self._parse_data_ir(inputs_ir)
            inputs_rgb = torch.cat((inputs_rgb, inputs_rgb1), 0)
            labels_rgb = torch.cat((labels_rgb, labels_rgb), -1)

            if stage == 2:
                # load data
                inputs_rgb2ir = data_loader_rgb2ir.next()
                inputs_ir2rgb = data_loader_ir2rgb.next()

                # process inputs
                inputs_rgb2ir, inputs_rgb2ir1, labels_rgb2ir, indexes_rgb2ir = self._parse_data_rgb(inputs_rgb2ir)
                inputs_ir2rgb, labels_ir2rgb, indexes_ir2rgb = self._parse_data_ir(inputs_ir2rgb)
                inputs_rgb2ir = torch.cat((inputs_rgb2ir, inputs_rgb2ir1), 0)
                labels_rgb2ir = torch.cat((labels_rgb2ir, labels_rgb2ir), -1)

            if stage == 1:
                # forward
                feat, f_dis = self._forward(inputs_rgb, inputs_ir, modal=0)
                feat_rgb, feat_ir = torch.split(feat, inputs_rgb.size(0))

                # loss function
                loss_rgb = self.memory_rgb(feat_rgb, labels_rgb)
                loss_ir = self.memory_ir(feat_ir, labels_ir)

                loss = loss_rgb + loss_ir

            else:
                # forward
                feat, f_dis = self._forward(inputs_rgb, inputs_ir, modal=0)
                feat_2, f_dis2 = self._forward(inputs_rgb2ir, inputs_ir2rgb, modal=0)
                feat_rgb, feat_ir = torch.split(feat, inputs_rgb.size(0))
                feat_rgb2, feat_ir2 = torch.split(feat_2, inputs_rgb2ir.size(0))

                dis_label = torch.cat((torch.ones(inputs_rgb.size(0)), torch.zeros(inputs_ir.size(0))), dim=0).cuda()
                dis_label2 = torch.cat((torch.ones(inputs_rgb2ir.size(0)), torch.zeros(inputs_ir2rgb.size(0))),
                                       dim=0).cuda()

                # loss function
                loss_rgb = self.memory_rgb(feat_rgb, labels_rgb)
                loss_ir = self.memory_ir(feat_ir, labels_ir)

                loss_dis = criterion_dis(f_dis.view(-1), dis_label) * 0.5 + criterion_dis(f_dis2.view(-1),
                                                                                          dis_label2) * 0.5
                loss_ir2rgb = self.memory_rgb(feat_ir2, labels_ir2rgb)
                loss_rgb2ir = self.memory_ir(feat_rgb2, labels_rgb2ir)

                loss = loss_rgb + loss_ir + loss_rgb2ir + loss_ir2rgb + loss_dis

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Loss rgb {:.3f}\tLoss ir {:.3f}\t'.format(epoch, i + 1, train_iters, batch_time.val,
                                                                 batch_time.avg, data_time.val, data_time.avg,
                                                                 losses.val, losses.avg, loss_rgb, loss_ir))

                if stage == 2:
                    print('Loss rgb2ir {:.3f}\tLoss ir2rgb {:.3f}\tLoss dis {:.3f}'.format(loss_rgb2ir, loss_ir2rgb,
                                                                                           loss_dis))

    def _parse_data_rgb(self, inputs):
        imgs, imgs1, _, pids, _, indexes = inputs
        return imgs.cuda(), imgs1.cuda(), pids.cuda(), indexes.cuda()

    def _parse_data_ir(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, x1, x2, modal=0):
        return self.encoder(x1, x2, modal=modal)
