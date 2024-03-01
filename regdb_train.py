# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import argparse
import collections
import os.path as osp
import random
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as data
from PIL import Image
from sklearn.cluster import DBSCAN
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from clustercontrast import datasets
from clustercontrast import models
from clustercontrast.evaluators import extract_features
from clustercontrast.models.cm import ClusterMemory
from clustercontrast.trainers import DualModalityShard_Trainer
from clustercontrast.utils.data import IterLoader
from clustercontrast.utils.data import transforms as T
from clustercontrast.utils.data.preprocessor import Preprocessor, Preprocessor_visible
from clustercontrast.utils.data.sampler import RandomMultipleGallerySampler, RandomMultipleGallerySamplerNoCam
from clustercontrast.utils.faiss_rerank import compute_jaccard_distance
from clustercontrast.utils.logging import Logger
from clustercontrast.utils.serialization import load_checkpoint, save_checkpoint
from utils import ChannelRandomErasing, ChannelAdapGray, ChannelExchange
from utils.CRLR import label_refinement


def get_data(name, data_dir, modal="rgb", mode='all', suffix="", trial=0):
    root = data_dir
    name = name + "_" + modal
    dataset = datasets.create(name, root, mode, suffix=suffix, trial=trial)
    return dataset


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transformer = T.Compose([T.Resize((height, width), interpolation=3),  # T.RandomGrayscale(p=1),
                                  T.ToTensor(), normalizer])
    if testset is None:
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
                             batch_size=batch_size, num_workers=workers, shuffle=False, pin_memory=True)

    return test_loader


def get_train_loader(dataset, height, width, batch_size, workers, num_instances, iters, trainset=None, no_cam=False,
                     mode='rgb'):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transformer_rgb = T.Compose(
        [T.Resize((height, width), interpolation=3), T.Pad(10), T.RandomCrop((height, width)),
         T.RandomHorizontalFlip(p=0.5),  # T.RandomGrayscale(p=0.1),
         T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5)])
    train_transformer_ir = T.Compose(
        [T.Resize((height, width), interpolation=3), T.Pad(10), T.RandomCrop((height, width)), T.RandomHorizontalFlip(),
         T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5), ChannelAdapGray(probability=0.5)])
    train_transformer_aug = T.Compose(
        [T.Resize((height, width), interpolation=3), T.Pad(10), T.RandomCrop((height, width)),
         T.RandomHorizontalFlip(p=0.5), T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
         T.ToTensor(), normalizer, ChannelRandomErasing(probability=0.5), ChannelExchange(gray=2)])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        if no_cam:
            sampler = RandomMultipleGallerySamplerNoCam(train_set, num_instances)
        else:
            sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    if mode == 'rgb':
        train_loader = IterLoader(DataLoader(
            Preprocessor_visible(train_set, root=dataset.images_dir, transform=train_transformer_rgb,
                                 transform_aug=train_transformer_aug), batch_size=batch_size, num_workers=workers,
            sampler=sampler, shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)
    else:
        train_loader = IterLoader(
            DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer_ir),
                       batch_size=batch_size, num_workers=workers, sampler=sampler, shuffle=not rmgs_flag,
                       pin_memory=True, drop_last=True), length=iters)

    return train_loader


def process_test_regdb(img_dir, trial=1, modal='visible'):
    if modal == 'visible':
        input_data_path = img_dir + 'idx/test_visible_{}'.format(trial) + '.txt'
    elif modal == 'thermal':
        input_data_path = img_dir + 'idx/test_thermal_{}'.format(trial) + '.txt'

    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [img_dir + '/' + s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, np.array(file_label)


class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def extract_query_feat(model, query_loader, nquery):
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net(input, input, 1)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net(flip_input, flip_input, 1)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            query_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()

            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return query_feat_fc


def extract_gall_feat(model, gall_loader, ngall):
    pool_dim = 2048
    net = model
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            flip_input = fliplr(input)
            input = Variable(input.cuda())
            feat_fc = net(input, input, 2)
            flip_input = Variable(flip_input.cuda())
            feat_fc_1 = net(flip_input, flip_input, 2)
            feature_fc = (feat_fc.detach() + feat_fc_1.detach()) / 2
            fnorm_fc = torch.norm(feature_fc, p=2, dim=1, keepdim=True)
            feature_fc = feature_fc.div(fnorm_fc.expand_as(feature_fc))
            gall_feat_fc[ptr:ptr + batch_num, :] = feature_fc.cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))
    return gall_feat_fc


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()  # N x C x H x W
    img_flip = img.index_select(3, inv_idx)
    return img_flip


def eval_regdb(distmat, q_pids, g_pids, max_rank=20):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.  # number of valid query

    # only two cameras
    q_camids = np.ones(num_q).astype(np.int32)
    g_camids = 2 * np.ones(num_g).astype(np.int32)

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        raw_cmc = matches[q_idx][keep]  # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = raw_cmc.cumsum()

        # compute mINP
        # refernece Deep Learning for Person Re-identification: A Survey and Outlook
        pos_idx = np.where(raw_cmc == 1)
        pos_max_idx = np.max(pos_idx)
        inp = cmc[pos_max_idx] / (pos_max_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)
    return all_cmc, mAP, mINP


def create_model(args):
    model = models.create(args.arch, num_features=args.features, norm=True, dropout=args.dropout, num_classes=0,
                          pooling_type=args.pooling_type)
    # use CUDA
    model.cuda()
    model = nn.DataParallel(model)  # ,output_device=1)
    return model


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    log_s1_name = 'regdb_s1'
    log_s2_name = 'regdb_s2'

    main_worker_stage1(args, log_s1_name)
    main_worker_stage2(args, log_s1_name, log_s2_name)


def main_worker_stage1(args, log_s1_name):
    logs_dir_root = osp.join(args.logs_dir + '/' + log_s1_name)
    trial = args.trial
    start_epoch = 0
    best_mAP = 0
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()
    cudnn.benchmark = True
    args.batch_size = 32
    args.num_instances = 4

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load dataset.")

    dataset_rgb = get_data(args.dataset, args.data_dir, mode=args.mode, modal="rgb", trial=trial)
    dataset_ir = get_data(args.dataset, args.data_dir, mode=args.mode, modal="ir", trial=trial)

    # Create model
    model = create_model(args)

    # Init optimizer
    params = [{"params": [value]} for _, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    trainer = DualModalityShard_Trainer(model)

    for epoch in range(args.epochs):
        with torch.no_grad():
            args.eps_ir = 0.3
            print('Thermal Cluster Criterion: eps: {:.3f}'.format(args.eps_ir))
            cluster_ir = DBSCAN(eps=args.eps_ir, min_samples=4, metric='precomputed', n_jobs=-1)
            args.eps_rgb = 0.3
            print('Visible Cluster Criterion: eps: {:.3f}'.format(args.eps_rgb))
            cluster_rgb = DBSCAN(eps=args.eps_rgb, min_samples=4, metric='precomputed', n_jobs=-1)

            # extract features of visible images
            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, 128, args.workers,
                                                 testset=sorted(dataset_rgb.train))
            print('==> Extracting RGB data features')
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            # extract features of infrared images
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width, 128, args.workers,
                                                testset=sorted(dataset_ir.train))
            print('==> Extracting IR data features')
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir

            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2)
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2)

            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)

            del rerank_dist_rgb, rerank_dist_ir

            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)

        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp, momentum=args.momentum,
                                   use_hard=args.use_hard).cuda()
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp, momentum=args.momentum,
                                  use_hard=args.use_hard).cuda()

        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()

        trainer.memory_rgb = memory_rgb
        trainer.memory_ir = memory_ir

        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        pseudo_labeled_dataset_ir = []
        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        train_loader_rgb = get_train_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers,
                                            args.num_instances, iters, trainset=pseudo_labeled_dataset_rgb,
                                            no_cam=args.no_cam, mode='rgb')
        train_loader_ir = get_train_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers,
                                           args.num_instances, iters, trainset=pseudo_labeled_dataset_ir,
                                           no_cam=args.no_cam, mode='ir')

        train_loader_rgb.new_epoch()
        train_loader_ir.new_epoch()

        trainer.train(epoch, data_loader_rgb=train_loader_rgb, data_loader_ir=train_loader_ir, optimizer=optimizer,
                      print_freq=args.print_freq, train_iters=len(train_loader_ir), stage=1)

        if epoch >= 0 and ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([T.ToPILImage(), T.Resize((args.height, args.width)), T.ToTensor(), normalize])
            mode = 'all'
            data_path = './data/regdb/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.width, args.height))
            gall_loader = data.DataLoader(gallset, batch_size=64, shuffle=False, num_workers=args.workers)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.width, args.height))
            query_loader = data.DataLoader(queryset, batch_size=64, shuffle=False, num_workers=args.workers)

            nquery = len(query_label)
            query_feat_fc = extract_query_feat(model, query_loader, nquery)

            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)

            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            #################################
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch + 1, 'best_mAP': best_mAP, }, is_best,
                            fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP,
                                                                                            ' *' if is_best else ''))

        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


def main_worker_stage2(args, log_s1_name, log_s2_name):
    logs_dir_root = osp.join('logs/' + log_s2_name)
    trial = args.trial
    start_epoch = 0
    best_mAP = 0
    args.logs_dir = osp.join(logs_dir_root, str(trial))
    start_time = time.monotonic()
    cudnn.benchmark = True
    args.batch_size = 16
    args.num_instances = 4

    sys.stdout = Logger(osp.join(args.logs_dir, str(trial) + 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters > 0) else None
    print("==> Load dataset.")
    dataset_rgb = get_data(args.dataset, args.data_dir, mode=args.mode, modal="rgb", trial=trial)
    dataset_ir = get_data(args.dataset, args.data_dir, mode=args.mode, modal="ir", trial=trial)

    # Create model
    model = create_model(args)
    checkpoint = load_checkpoint(osp.join('./logs/' + log_s1_name + '/' + str(trial), 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # Init optimizer
    params = [{"params": [value]} for key, value in model.named_parameters() if value.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    trainer = DualModalityShard_Trainer(model)
    for epoch in range(args.epochs):
        with torch.no_grad():
            if epoch == 0:
                args.eps_ir = 0.3
                print('Thermal Cluster Criterion: eps: {:.3f}'.format(args.eps_ir))
                cluster_ir = DBSCAN(eps=args.eps_ir, min_samples=4, metric='precomputed', n_jobs=-1)
                args.eps_rgb = 0.3
                print('Visible Cluster Criterion: eps: {:.3f}'.format(args.eps_rgb))
                cluster_rgb = DBSCAN(eps=args.eps_rgb, min_samples=4, metric='precomputed', n_jobs=-1)

            # extract features of visible images
            print('==> Create pseudo labels for unlabeled RGB data')
            cluster_loader_rgb = get_test_loader(dataset_rgb, args.height, args.width, 256, args.workers,
                                                 testset=sorted(dataset_rgb.train))
            print('==> Extracting RGB data features')
            features_rgb, _ = extract_features(model, cluster_loader_rgb, print_freq=50, mode=1)
            del cluster_loader_rgb
            features_rgb = torch.cat([features_rgb[f].unsqueeze(0) for f, _, _ in sorted(dataset_rgb.train)], 0)

            # extract features of infrared images
            print('==> Create pseudo labels for unlabeled IR data')
            cluster_loader_ir = get_test_loader(dataset_ir, args.height, args.width, 256, args.workers,
                                                testset=sorted(dataset_ir.train))
            print('==> Extracting IR data features')
            features_ir, _ = extract_features(model, cluster_loader_ir, print_freq=50, mode=2)
            del cluster_loader_ir

            features_ir = torch.cat([features_ir[f].unsqueeze(0) for f, _, _ in sorted(dataset_ir.train)], 0)

            rerank_dist_rgb = compute_jaccard_distance(features_rgb, k1=args.k1, k2=args.k2)
            rerank_dist_ir = compute_jaccard_distance(features_ir, k1=args.k1, k2=args.k2)

            pseudo_labels_rgb = cluster_rgb.fit_predict(rerank_dist_rgb)
            pseudo_labels_ir = cluster_ir.fit_predict(rerank_dist_ir)
            del rerank_dist_rgb, rerank_dist_ir

            num_cluster_rgb = len(set(pseudo_labels_rgb)) - (1 if -1 in pseudo_labels_rgb else 0)
            num_cluster_ir = len(set(pseudo_labels_ir)) - (1 if -1 in pseudo_labels_ir else 0)

        # generate new dataset and calculate cluster centers
        @torch.no_grad()
        def generate_cluster_features(labels, features):
            centers = collections.defaultdict(list)
            for i, label in enumerate(labels):
                if label == -1:
                    continue
                centers[labels[i]].append(features[i])

            centers = [torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())]

            centers = torch.stack(centers, dim=0)
            return centers

        cluster_features_rgb = generate_cluster_features(pseudo_labels_rgb, features_rgb)
        cluster_features_ir = generate_cluster_features(pseudo_labels_ir, features_ir)

        cluster_features_rgb = F.normalize(cluster_features_rgb, dim=1)
        cluster_features_ir = F.normalize(cluster_features_ir, dim=1)

        print('==> Cluster Similarity Matching.')

        sim_matrix = np.zeros((num_cluster_rgb, num_cluster_ir))

        for index_rgb in tqdm(range(num_cluster_rgb)):
            for index_ir in range(num_cluster_ir):
                rgbs = np.where(pseudo_labels_rgb == index_rgb)[0]
                irs = np.where(pseudo_labels_ir == index_ir)[0]
                feat_rgbs = features_rgb[rgbs]
                feat_irs = features_ir[irs]
                value = torch.matmul(feat_rgbs, feat_irs.t()).numpy().mean().mean()
                sim_matrix[index_rgb][index_ir] = value

        rgb2ir = np.argmax(sim_matrix, axis=1)
        ir2rgb = np.argmax(sim_matrix, axis=0)

        print('==> Cluster Relationship based Label Refinement.')
        pseudo_labels_ir_new, pseudo_labels_ir2rgb, pseudo_labels_rgb_split = label_refinement(num_cluster_rgb,
                                                                                               pseudo_labels_rgb,
                                                                                               pseudo_labels_ir, ir2rgb,
                                                                                               features_rgb,
                                                                                               cluster_features_ir,
                                                                                               cluster_features_rgb,
                                                                                               threshold=args.threshold)
        pseudo_labels_rgb_new, pseudo_labels_rgb2ir, pseudo_labels_ir_split = label_refinement(num_cluster_ir,
                                                                                               pseudo_labels_ir,
                                                                                               pseudo_labels_rgb,
                                                                                               rgb2ir, features_ir,
                                                                                               cluster_features_rgb,
                                                                                               cluster_features_ir,
                                                                                               threshold=args.threshold)

        num_cluster_rgb = len(set(pseudo_labels_rgb_split)) - (1 if -1 in pseudo_labels_rgb_split else 0)
        num_cluster_ir = len(set(pseudo_labels_ir_split)) - (1 if -1 in pseudo_labels_ir_split else 0)

        # Weighted Modality-shared Memory
        @torch.no_grad()
        def generate_cluster_features_hat(cluster_feature, features1, labels1, features2, labels2):
            w1 = collections.defaultdict(list)
            for i, label in enumerate(labels1):
                if label == -1:
                    continue
                w1[label].append(features1[i])

            w2 = collections.defaultdict(list)
            for i, label in enumerate(labels2):
                if label == -1:
                    continue
                w2[label].append(features2[i])

            weights1 = collections.defaultdict(list)
            for label in np.unique(labels1):
                if label == -1:
                    continue
                features_label = torch.stack(w1[label], dim=0)
                weight = -torch.matmul(cluster_feature[label], features_label.t()) / args.temp_dyn
                weights1[label] = F.softmax(weight, dim=0)

            weights2 = collections.defaultdict(list)
            for label in np.unique(labels1):
                if label == -1:
                    continue
                if len(w2[label]) != 0:
                    features_label = torch.stack(w2[label], dim=0)
                    weight = -torch.matmul(cluster_feature[label], features_label.t()) / args.temp_dyn
                    weights2[label] = F.softmax(weight, dim=0)

            centers1 = collections.defaultdict(list)
            for i, label in enumerate(labels1):
                if label == -1:
                    continue
                centers1[label].append(features1[i] * weights1[label][len(centers1[label])])

            centers2 = collections.defaultdict(list)
            for i, label in enumerate(labels2):
                if label == -1:
                    continue
                centers2[label].append(features2[i] * weights2[label][len(centers2[label])])
            centers = []
            num_cluster = len(centers1)

            for i in range(num_cluster):
                if len(centers2[i]) == 0:
                    centers.append(torch.stack(centers1[i], dim=0).sum(0))
                else:
                    centers.append(
                        (torch.stack(centers1[i], dim=0).sum(0) + torch.stack(centers2[i], dim=0).sum(0)) / 2.0)
            centers = torch.stack(centers, dim=0)
            return centers

        @torch.no_grad()
        def generate_cluster_features_weighted(features1, features2, labels1, labels2):
            centers1 = collections.defaultdict(list)
            centers2 = collections.defaultdict(list)
            for i, label in enumerate(labels1):
                if label == -1:
                    continue
                centers1[label].append(features1[i])

            for i, label in enumerate(labels2):
                if label == -1:
                    continue
                centers2[label].append(features2[i])
            num_cluster = len(centers1)
            centers = []
            for i in range(num_cluster):
                if len(centers2[i]) == 0:
                    centers.append(torch.stack(centers1[i], dim=0).mean(0))
                else:
                    centers.append(
                        (torch.stack(centers1[i], dim=0).mean(0) + torch.stack(centers2[i], dim=0).mean(0)) / 2.0)
            centers = torch.stack(centers, dim=0)
            return centers

        # balance the difference in sample numbers across modalities
        cluster_features_rgb = generate_cluster_features_weighted(features_rgb, features_ir, pseudo_labels_rgb_split,
                                                                  pseudo_labels_ir2rgb)
        cluster_features_ir = generate_cluster_features_weighted(features_ir, features_rgb, pseudo_labels_ir_split,
                                                                 pseudo_labels_rgb2ir)

        cluster_features_rgb_hat = generate_cluster_features_hat(cluster_features_rgb, features_rgb,
                                                                 pseudo_labels_rgb_split, features_ir,
                                                                 pseudo_labels_ir2rgb)
        cluster_features_ir_hat = generate_cluster_features_hat(cluster_features_ir, features_ir,
                                                                pseudo_labels_ir_split, features_rgb,
                                                                pseudo_labels_rgb2ir)

        cluster_features_rgb = cluster_features_rgb * args.gamma + cluster_features_rgb_hat * (1 - args.gamma)
        cluster_features_ir = cluster_features_ir * args.gamma + cluster_features_ir_hat * (1 - args.gamma)

        memory_rgb = ClusterMemory(model.module.num_features, num_cluster_rgb, temp=args.temp, momentum=args.momentum,
                                   use_hard=args.use_hard).cuda()
        memory_ir = ClusterMemory(model.module.num_features, num_cluster_ir, temp=args.temp, momentum=args.momentum,
                                  use_hard=args.use_hard).cuda()

        memory_rgb.features = F.normalize(cluster_features_rgb, dim=1).cuda()
        memory_ir.features = F.normalize(cluster_features_ir, dim=1).cuda()
        del features_rgb, features_ir

        trainer.memory_rgb = memory_rgb
        trainer.memory_ir = memory_ir
        # Dual-Modality-Shared Contrastive Learning
        pseudo_labeled_dataset_rgb = []
        rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb)):
            if label != -1:
                pseudo_labeled_dataset_rgb.append((fname, label.item(), cid))
                rgb_label.append(label.item())
        print('==> Statistics for RGB epoch {}: {} clusters'.format(epoch, num_cluster_rgb))

        pseudo_labeled_dataset_ir = []
        ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir)):
            if label != -1:
                pseudo_labeled_dataset_ir.append((fname, label.item(), cid))
                ir_label.append(label.item())
        print('==> Statistics for IR epoch {}: {} clusters'.format(epoch, num_cluster_ir))

        pseudo_labeled_dataset_rgb2ir = []
        rgb2ir_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_rgb.train), pseudo_labels_rgb2ir)):
            if label != -1:
                pseudo_labeled_dataset_rgb2ir.append((fname, label.item(), cid))
                rgb2ir_label.append(label.item())

        pseudo_labeled_dataset_ir2rgb = []
        ir2rgb_label = []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_ir.train), pseudo_labels_ir2rgb)):
            if label != -1:
                pseudo_labeled_dataset_ir2rgb.append((fname, label.item(), cid))
                ir2rgb_label.append(label.item())

        train_loader_rgb = get_train_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers,
                                            args.num_instances, iters, trainset=pseudo_labeled_dataset_rgb,
                                            no_cam=args.no_cam, mode='rgb')
        train_loader_ir = get_train_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers,
                                           args.num_instances, iters, trainset=pseudo_labeled_dataset_ir,
                                           no_cam=args.no_cam, mode='ir')
        train_loader_rgb2ir = get_train_loader(dataset_rgb, args.height, args.width, args.batch_size, args.workers,
                                               args.num_instances, iters, trainset=pseudo_labeled_dataset_rgb2ir,
                                               no_cam=args.no_cam, mode='rgb')
        train_loader_ir2rgb = get_train_loader(dataset_ir, args.height, args.width, args.batch_size, args.workers,
                                               args.num_instances, iters, trainset=pseudo_labeled_dataset_ir2rgb,
                                               no_cam=args.no_cam, mode='ir')

        train_loader_rgb.new_epoch()
        train_loader_ir.new_epoch()
        train_loader_rgb2ir.new_epoch()
        train_loader_ir2rgb.new_epoch()

        trainer.train(epoch, train_loader_rgb, train_loader_ir, train_loader_rgb2ir, train_loader_ir2rgb,
                      optimizer=optimizer, print_freq=args.print_freq, train_iters=len(train_loader_ir), stage=2)

        if epoch >= 0 and ((epoch + 1) % args.eval_step == 0 or (epoch == args.epochs - 1)):
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            transform_test = T.Compose([T.ToPILImage(), T.Resize((args.height, args.width)), T.ToTensor(), normalize])
            mode = 'all'
            data_path = './data/regdb/'
            query_img, query_label = process_test_regdb(data_path, trial=trial, modal='visible')
            gall_img, gall_label = process_test_regdb(data_path, trial=trial, modal='thermal')

            gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.width, args.height))
            gall_loader = data.DataLoader(gallset, batch_size=64, shuffle=False, num_workers=args.workers)
            queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.width, args.height))
            query_loader = data.DataLoader(queryset, batch_size=64, shuffle=False, num_workers=args.workers)

            nquery = len(query_label)
            query_feat_fc = extract_query_feat(model, query_loader, nquery)

            ngall = len(gall_label)
            gall_feat_fc = extract_gall_feat(model, gall_loader, ngall)

            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)

            print('Test Trial: {}'.format(trial))
            print(
                'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                    cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))

            #################################
            is_best = (mAP > best_mAP)
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({'state_dict': model.state_dict(), 'epoch': epoch + 1, 'best_mAP': best_mAP, }, is_best,
                            fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  model mAP: {:5.1%}  best: {:5.1%}{}\n'.format(epoch, mAP, best_mAP,
                                                                                            ' *' if is_best else ''))

        lr_scheduler.step()
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Unsupervised Visible-Infrared Person Re-Identification via Dual-Modality-Shared Modality-Shared Learning and Label Refinement")

    # data
    parser.add_argument('-d', '--dataset', type=str, default='regdb', choices=datasets.names())
    parser.add_argument('-b', '--batch_size', type=int, default=8)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=288, help="input height")
    parser.add_argument('--width', type=int, default=144, help="input width")
    parser.add_argument('--num_instances', type=int, default=8,
                        help="each mini-batch consist of (batch_size // num_instances) identities, and each identity has num_instances instances, default: 0 (NOT USE)")
    parser.add_argument('--mode', type=str, default='v2t', help='all or indoor (sysu test), t2v or v2t (regdb test)')
    parser.add_argument('--shot', default=1, type=int, help='1 for single shot;10 for multi shot')
    parser.add_argument('--trial', default=1, type=int, help='trial (only for RegDB dataset)')

    # cluster
    parser.add_argument('--eps_ir', type=float, default=0.3,
                        help="max neighbor distance of thermal modality for DBSCAN")
    parser.add_argument('--eps_rgb', type=float, default=0.3,
                        help="max neighbor distance of visible modality for DBSCAN")
    parser.add_argument('--eps_sh', type=float, default=0.3, help="max neighbor distance of shared modality for DBSCAN")
    parser.add_argument('--eps_gap', type=float, default=0.02,
                        help="multi-scale criterion for measuring cluster reliability")
    parser.add_argument('--k1', type=int, default=30, help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6, help="hyperparameter for jaccard distance")

    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet', choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--pooling-type', type=str, default='gem')
    parser.add_argument('--momentum', type=float, default=0.1, help="update momentum for the hybrid memory")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--temp_dyn', type=float, default=0.09)
    parser.add_argument('--threshold', type=float, default=0.7)
    parser.add_argument('--temp', type=float, default=0.05, help="temperature for scaling contrastive loss")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'logs'))
    parser.add_argument('--use-hard', action="store_true")
    parser.add_argument('--no-cam', action="store_true")

    main()
