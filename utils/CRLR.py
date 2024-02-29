import collections

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


def label_refinement(num_cluster_rgb, pseudo_label_rgb, pseudo_label_ir, relation_ir2rgb, features_rgb,
                     cluster_features_ir, cluster_features_rgb, threshold=0.6):
    '''
    given relationship from ir to rgb, the module give the merged ir labels, splited rgb labels and the refined relationship from ir to rgb
    '''

    pseudo_labels_rgb = pseudo_label_rgb.copy()
    pseudo_labels_ir = pseudo_label_ir.copy()
    features = features_rgb.clone()
    cluster_features = cluster_features_ir.clone()
    cluster_features_rgb = cluster_features_rgb.clone()

    ir2rgb = collections.defaultdict(list)  # record each ir cluster matching to the rgb cluster
    for index, item in enumerate(relation_ir2rgb):
        ir2rgb[item].append(index)
    new_map = collections.defaultdict(list)
    newlabel_ir = 0  # record the updated ir label
    add_label = num_cluster_rgb
    add_map = collections.defaultdict(list)
    tranced_index = np.zeros(len(pseudo_labels_rgb))
    pseudo_labels_rgb_split = pseudo_labels_rgb.copy()
    for key in ir2rgb.keys():
        cluster_count = 0
        tmp_list = ir2rgb[key]
        tmp_list1 = []
        # split
        if len(tmp_list) == 1:  # if only one ir cluster matches to the rgb cluster
            new_map[tmp_list[0]] = newlabel_ir
            tmp_list1.append(tmp_list[0])
            newlabel_ir = newlabel_ir + 1
            cluster_count = cluster_count + 1
        else:  # multiple ir clusters match to the same rgb cluster
            for index in range(len(tmp_list)):
                if index == 0:
                    new_map[tmp_list[0]] = newlabel_ir
                    newlabel_ir = newlabel_ir + 1
                    tmp_list1.append(tmp_list[0])
                    cluster_count = cluster_count + 1
                else:
                    flag = True
                    for index1 in range(index):
                        if float(torch.matmul(cluster_features[tmp_list[index]], cluster_features[tmp_list[
                            index1]].t())) > threshold:  # decide to merge ir clusters if the similarity between them exceeds the threshold
                            new_map[tmp_list[index]] = new_map[tmp_list[index1]]
                            flag = False
                            cluster_flag = cluster_features_rgb[key].view(1, -1)
                            cluster_feats = torch.cat((cluster_features[tmp_list[index]].view(1, -1),
                                                       cluster_features[tmp_list[index1]].view(1, -1)), dim=0)
                            ind = torch.argmax(torch.matmul(cluster_feats, cluster_flag.t()))
                            cluster_features[tmp_list[index1]] = cluster_feats[ind]
                            break
                    if flag:
                        new_map[tmp_list[index]] = newlabel_ir
                        newlabel_ir = newlabel_ir + 1
                        cluster_count = cluster_count + 1
                        tmp_list1.append(tmp_list[index])
        if cluster_count != 1:  # if the number of ir clusters matching to the visible cluster exceeds one
            cluster_feat = cluster_features[tmp_list1]
            tmp_list = list(range(newlabel_ir - cluster_count, newlabel_ir))
            instance_indexs = np.where(pseudo_labels_rgb == key)[0]
            instance_indexs = instance_indexs[tranced_index[instance_indexs] == 0]
            tranced_index[instance_indexs] = 1
            new_labels = [key] + list(range(add_label, add_label + cluster_count - 1))

            instance_feat = features[instance_indexs]
            instance_count = torch.unique(instance_feat, dim=0).size(0)

            # split the rgb cluster to k parts by k-means, k is the number of matched ir clusters
            if instance_count <= cluster_count:
                cluster = KMeans(n_clusters=instance_count).fit(instance_feat)
            else:
                cluster = KMeans(n_clusters=cluster_count).fit(instance_feat)

            centroid = torch.from_numpy(cluster.cluster_centers_).to(torch.float32)
            centroid = F.normalize(centroid, dim=1)
            sim_centroid2cluster = torch.matmul(centroid, cluster_feat.t())
            sim_rank = torch.argsort(sim_centroid2cluster, dim=0,
                                     descending=True)  # rematch the rgb clusters after splitting to merged ir clusters according to the similarities between cluster centers
            selected = []
            y_pred = cluster.labels_

            for selects in range(cluster_count):
                delivered = False
                if selects < len(set(y_pred)):
                    for rank in sim_rank[:, selects]:
                        if int(rank) not in selected:
                            selected.append(int(rank))
                            delivered = True
                            break
                if not delivered and len(set(y_pred)) < cluster_count:
                    selected.append(int(sim_rank[:, selects][0]))

            for selects in range(cluster_count):
                pseudo_labels_rgb_split[instance_indexs[y_pred == selects]] = new_labels[selects]
                add_map[tmp_list[selects]] = new_labels[selected[selects]]
        else:
            add_map[newlabel_ir - 1] = key

    # update the cross-modality relationship
    pseudo_labels_ir_new = np.zeros(len(pseudo_labels_ir))
    pseudo_labels_ir2rgb = np.zeros(len(pseudo_labels_ir))
    for ind, label in enumerate(pseudo_labels_ir):
        if label == -1:
            pseudo_labels_ir_new[ind] = -1
            pseudo_labels_ir2rgb[ind] = -1
            continue
        pseudo_labels_ir_new[ind] = new_map[label]
        pseudo_labels_ir2rgb[ind] = add_map[pseudo_labels_ir_new[ind]]
    return pseudo_labels_ir_new.astype(int), pseudo_labels_ir2rgb.astype(int), pseudo_labels_rgb_split
