"""
multi-domain transfer CLIP under KRR version (dual space)
"""

import clip
import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import yaml
from easydict import EasyDict
from tqdm import tqdm

from scenario_datasets import build_dataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *

# DIR_PATH = os.path.dirname(os.path.realpath(__file__))
DIR_PATH = '/HDD/Suvam_Backup/Suvam/DATA'

cfg_file = "configs/analytic_clip.yaml"
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
cfg = EasyDict(cfg)

seed = cfg.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset_sequence = cfg.datasets
print("Multi-task dataset sequence: ", dataset_sequence)

merged_classnames = []  # for clip zero-shot
for _, train_dataset in enumerate(dataset_sequence):
    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'CIFAR100'))
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'MNIST'))
    else:
        dataset = build_dataset(train_dataset, os.path.join(DIR_PATH), cfg.num_shots)
    merged_classnames += dataset.classnames

fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top3_acc_table  = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top5_acc_table  = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top3_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top5_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_MRR_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_cos_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))

in_domain_acc_list = []

cfg.previous_class_num = 0
current_class_names = []

"""
Loading model
"""
print('Loading pretrained CLIP model...')
clip_model, train_transform, val_preprocess = clip.load(cfg.backbone, device=cfg.device, jit=False)
template = ['a photo of a {}.']
clip_model.eval()
krr = kernel_ridge_regression(lamda=0.001, gamma=cfg.gamma)

cfg.trained_class_num = 0
feature_memory = None
y = None

"""
Training on dataset sequence
"""
for task_id, train_dataset in enumerate(dataset_sequence):
    print(f"------------------ Start training on task-{task_id + 1}: dataset-{train_dataset}. ---------------------")

    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                           batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'CIFAR100'))
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                        batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'MNIST'))
    else:
        dataset = build_dataset(train_dataset, os.path.join(DIR_PATH), cfg.num_shots)

    current_class_names += dataset.classnames
    cfg.increment = len(dataset.classnames)
    cfg.current_class_num = len(current_class_names)

    if train_dataset == "cifar100" or train_dataset == "mnist":
        train_loader = dataset.train_loader
    else:
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.batch_size, tfm=train_transform,
                                         is_train=True, shuffle=True, augmentation_time=cfg.augmentation_time)

    current_train_features = []
    current_train_one_hot_labels = []
    with torch.no_grad():
        for i, (images, target) in \
                enumerate(tqdm(train_loader, desc=f'Extracting training features', total=len(train_loader),
                               unit='batch')):
            target += cfg.trained_class_num
            images, target = images.to(cfg.device), target.to(cfg.device)
            img_embeddings = encode_images(clip_model, images)

            train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

            current_train_features.append(img_embeddings)
            current_train_one_hot_labels.append(train_labels_one_hot)

    current_train_features = torch.cat(current_train_features, dim=0)
    current_train_one_hot_labels = torch.cat(current_train_one_hot_labels, dim=0).cpu().numpy()

    if task_id == 0:
        feature_memory = current_train_features
        y = current_train_one_hot_labels
    else:
        feature_memory = torch.cat([feature_memory, current_train_features], dim=0)
        y = np.concatenate([y, np.zeros((y.shape[0], cfg.increment))], axis=1)
        y = np.concatenate([y, current_train_one_hot_labels], axis=0)

    alpha = krr.train(feature_memory, y)  # obtain the dual parameter

    cfg.trained_class_num = cfg.current_class_num
    """
    Testing stage: test on every dataset (both trained & untrained) after training on each dataset
    """
    if cfg.eval_last and task_id < len(dataset_sequence) - 1:
        continue

    tested_cls_num = 0
    for test_id, test_dataset in enumerate(dataset_sequence):
        if cfg.eval_adapter and test_id > task_id:
            continue

        print(f"Evaluating on dataset-{test_id + 1}: {test_dataset}")

        if test_dataset == "cifar100":
            test_set = CIFAR100(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'CIFAR100'))
        elif test_dataset == "mnist":
            test_set = MNIST(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, data_root = os.path.join(DIR_PATH,'MNIST'))
        else:
            test_set = build_dataset(test_dataset, os.path.join(DIR_PATH), cfg.num_shots)

        if test_dataset == "cifar100" or test_dataset == "mnist":
            test_loader = test_set.test_loader
        else:
            test_loader = build_data_loader(data_source=test_set.test, batch_size=cfg.batch_size, is_train=False,
                                            tfm=val_preprocess, shuffle=False)

        clip_weights = clip_classifier(merged_classnames, template, clip_model)
        class_range_min, class_range_max = tested_cls_num, tested_cls_num + len(test_set.classnames)
        in_domain, in_domain_acc = 0.0, 0.0
        adapter_in_domain, adapter_in_domain_acc = 0.0, 0.0

        top1, top3, top5, test_num = 0.0, 0.0, 0.0, 0.0
        fusion_top1, fusion_top3, fusion_top5 = 0.0, 0.0, 0.0
        top3_score, top5_score = 0.0, 0.0
        fusion_top3_score, fusion_top5_score = 0.0, 0.0
        adapt_top1, adapt_top3, adapt_top5 = 0.0, 0.0, 0.0
        top_MRR_score = 0.0
        fusion_MRR_score = 0.0
        top_cos_score = 0.0
        fusion_cos_score = 0.0

        with torch.no_grad():
            for inputs, targets in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}',
                                        total=len(test_loader), unit='batch'):
                test_num += inputs.size(0)

                inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
                targets += tested_cls_num

                test_features = encode_images(clip_model, inputs)
                outputs = 100. * test_features @ clip_weights
                outputs = F.softmax(outputs, dim=-1)

                predict_cls = torch.argmax(outputs, dim=-1)

                # Zero-shot acc
                zs_outputs = outputs
                acc1, acc3, acc5 = cls_acc(zs_outputs, targets, topk=(1, 3, 5))
                score_3, score_5 = ranked_acc(zs_outputs, targets, topk=(3, 5), classnames=merged_classnames)
                mrr_score = MRR_Acc(zs_outputs, targets, classnames=merged_classnames)
                # breakpoint()
                top1 += acc1
                top3 += acc3
                top5 += acc5
                top3_score += score_3
                top5_score += score_5
                top_MRR_score += mrr_score
                top_cos_score += Cosine_Score(zs_outputs, targets, classnames=merged_classnames)
                # breakpoint()
                # Select samples that belong to learned domains determined by CLIP zero-shot
                mask = predict_cls < cfg.current_class_num
                if torch.sum(mask) > 0:
                    samples_to_adapt = test_features[mask]
                    outputs_adapted = krr.predict(samples_to_adapt, feature_memory)
                    outputs_adapted = torch.tensor(outputs_adapted, device=cfg.device, dtype=torch.float)
                    # Zero-padding to (N_ad, C_all)
                    padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                    outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)

                    outputs[mask] = (1-cfg.fusion_weight) * outputs[mask] + cfg.fusion_weight * outputs_adapted

                fusion_acc1, fusion_acc3, fusion_acc5 = cls_acc(outputs, targets, topk=(1, 3, 5))
                fusion_top1 += fusion_acc1
                fusion_top3 += fusion_acc3
                fusion_top5 += fusion_acc5
                # breakpoint()
                fusion_score_3, fusion_score_5 = ranked_acc(outputs, targets, topk=(3, 5), classnames=merged_classnames)
                fusion_top3_score += fusion_score_3
                fusion_top5_score += fusion_score_5
                mrr_score = MRR_Acc(outputs, targets, classnames=merged_classnames)
                # breakpoint()
                fusion_MRR_score += mrr_score
                fusion_cos_score += Cosine_Score(outputs, targets, classnames=merged_classnames)
                # breakpoint()
                # fusion_cos_score += fusion_cos_score
                if test_id <= task_id:
                    outputs = krr.predict(test_features, feature_memory)  # (N_ad, C_adapter)
                    outputs = torch.tensor(outputs, device=cfg.device, dtype=torch.float)
                    adapt_acc1 = cls_acc(outputs, targets)
                    adapt_top1 += adapt_acc1[0]

            top1, top3, top5 = (top1 / test_num) * 100, (top3 / test_num) * 100, (top5 / test_num) * 100
            print(f"Zero-shot top-1 acc for dataset-{test_id + 1}: {test_dataset}: {top1}")
            print(f"Zero-shot top-3 acc for dataset-{test_id + 1}: {test_dataset}: {top3}")
            print(f"Zero-shot top-5 acc for dataset-{test_id + 1}: {test_dataset}: {top5}")

            fusion_acc = (fusion_top1 / test_num) * 100
            fusion_top3_acc = (fusion_top3 / test_num) * 100
            fusion_top5_acc = (fusion_top5 / test_num) * 100
            fusion_top3_score_avg = (fusion_top3_score / test_num) * 100
            fusion_top5_score_avg = (fusion_top5_score / test_num) * 100
            fusion_MRR_score_avg = (fusion_MRR_score / test_num)
            fusion_cos_score_avg = (fusion_cos_score / test_num)
            # breakpoint()
            print(f"***** Fusion top-1 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_acc} *****")
            fusion_acc_table[task_id, test_id] = fusion_acc
            print(f"***** Fusion top-3 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_top3_acc} *****")
            print(f"***** Fusion top-5 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_top5_acc} *****")
            print(f"***** Fusion top-3 score for dataset-{test_id + 1}: {test_dataset}: {fusion_top3_score_avg} *****")
            print(f"***** Fusion top-5 score for dataset-{test_id + 1}: {test_dataset}: {fusion_top5_score_avg} *****")
            print(f"***** Fusion MRR score for dataset-{test_id + 1}: {test_dataset}: {fusion_MRR_score_avg} *****")
            print(f"***** Fusion Cosine Similarity score for dataset-{test_id + 1}: {test_dataset}: {fusion_cos_score_avg} *****")
            fusion_top3_acc_table[task_id, test_id] = fusion_top3_acc
            fusion_top5_acc_table[task_id, test_id] = fusion_top5_acc
            fusion_top3_score_table[task_id, test_id] = fusion_top3_score_avg
            fusion_top5_score_table[task_id, test_id] = fusion_top5_score_avg
            fusion_MRR_score_table[task_id, test_id] = fusion_MRR_score_avg
            fusion_cos_score_table[task_id, test_id] = fusion_cos_score_avg
            tested_cls_num += len(test_set.classnames)

upper_triangle_no_diag = np.triu(fusion_acc_table, k=1)
masked_matrix = np.ma.masked_equal(upper_triangle_no_diag, 0)
transfer_acc = np.mean(masked_matrix, axis=0)
transfer_avg_acc = np.mean(transfer_acc)
avg_acc = np.mean(fusion_acc_table, axis=0)
avg_avg_acc = np.mean(avg_acc)
print('average transfer acc: ', transfer_avg_acc)
print('average average acc: ', avg_avg_acc)
print('average last acc: ', np.mean(fusion_acc_table[-1, :]))
print('average last top-3 acc: ', np.mean(fusion_top3_acc_table[-1, :]))
print('average last top-5 acc: ', np.mean(fusion_top5_acc_table[-1, :]))
print('average last top-3 score: ', np.mean(fusion_top3_score_table[-1, :]))
print('average last top-5 score: ', np.mean(fusion_top5_score_table[-1, :]))
print("\n===== Final Evaluation Tables =====")

print("\nFusion Accuracy Table:")
print(fusion_acc_table)

print("\nFusion Top-3 Accuracy Table:")
print(fusion_top3_acc_table)

print("\nFusion Top-5 Accuracy Table:")
print(fusion_top5_acc_table)

print("\nFusion Top-3 Ranked Score Table:")
print(fusion_top3_score_table)

print("\nFusion Top-5 Ranked Score Table:")
print(fusion_top5_score_table)

print("\nFusion MRR Score Table:")
print(fusion_MRR_score_table)

print("\nFusion Cosine Similarity Score Table:")
print(fusion_cos_score_table)