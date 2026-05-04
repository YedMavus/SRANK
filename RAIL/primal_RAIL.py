import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import random
import numpy as np
import os
import yaml
from easydict import EasyDict

from scenario_datasets import build_dataset
from scenario_datasets.merged_dataset import MergedDataset
from scenario_datasets.utils import build_data_loader
from scenario_datasets.collections import CIFAR100, MNIST
from utils import *
import torchvision.utils as vutils
import random

DIR_PATH = '/HDD/Suvam_Backup/Suvam/DATA'


import json
import torchvision.utils as vutils

@torch.no_grad()
def generate_human_eval_set_from_sequence(
    dataset_sequence,
    continual_clip_adaptor,
    merged_classnames,
    rankings_path,
    device,
    save_dir,
    batch_size,
    val_preprocess,
    data_root,
    images_per_dataset=50,
    topk=3,
    template=("a photo of a {}.",)
):
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "human_eval.json")

    with open(rankings_path) as f:
        rankings = json.load(f)

    # Global CLIP classifier (C_all)
    clip_weights = clip_classifier(
        merged_classnames,
        list(template),
        continual_clip_adaptor.clip_model,
        device=device
    )

    meta = {}
    global_id = 0
    class_offset = 0  # used ONLY for prediction interpretation

    for dataset_name in dataset_sequence:
        collected = 0

        # -------- build dataset + loader --------
        if dataset_name == "cifar100":
            test_set = CIFAR100(
                num_shots=-1,
                preprocess=None,
                val_transform=val_preprocess,
                batch_size=batch_size,
                data_root=os.path.join(data_root, "CIFAR100")
            )
            test_loader = test_set.test_loader

        elif dataset_name == "mnist":
            test_set = MNIST(
                num_shots=-1,
                preprocess=None,
                val_transform=val_preprocess,
                batch_size=batch_size,
                data_root=os.path.join(data_root, "MNIST")
            )
            test_loader = test_set.test_loader

        else:
            test_set = build_dataset(dataset_name, data_root, -1)
            test_loader = build_data_loader(
                data_source=test_set.test,
                batch_size=batch_size,
                is_train=False,
                tfm=val_preprocess,
                shuffle=False
            )
        # ---------------------------------------

        num_classes = len(test_set.classnames)

        for inputs, targets in test_loader:
            if collected >= images_per_dataset:
                break

            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = continual_clip_adaptor.zero_shot(inputs, clip_weights)
            probs = F.softmax(outputs, dim=-1)
            _, pred_idx = probs.topk(topk, dim=1, largest=True, sorted=True)

            B = inputs.size(0)
            order = list(range(B))
            random.shuffle(order)

            for i in order:
                if collected >= images_per_dataset:
                    break

                img = inputs[i].cpu()

                # -------- CORRECT GT RANK LOGIC --------
                target_local_idx = targets[i].item()
                gt_name = test_set.classnames[target_local_idx]
                gt_rank_list = rankings[gt_name][:topk]
                # -------------------------------------

                # Global predictions
                pred_classes = [
                    merged_classnames[j] for j in pred_idx[i].tolist()
                ]

                img_name = f"{global_id:04d}.png"
                img_path = os.path.join(save_dir, img_name)

                vutils.save_image(img, img_path, normalize=True)

                meta[img_path] = {
                    "dataset": dataset_name,
                    "GT_ranks": [
                        {"class": x["class"], "rank": x["rank"]}
                        for x in gt_rank_list
                    ],
                    "pred_ranks": [
                        {"class": c, "rank": r + 1}
                        for r, c in enumerate(pred_classes)
                    ]
                }

                global_id += 1
                collected += 1

        class_offset += num_classes  # kept for conceptual clarity

    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)

    return global_id



class continual_clip_adaptor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.clip_model, self.train_preprocess, self.val_preprocess = clip.load(cfg.backbone, device=cfg.device, jit=False)
        self.feature_dim = 512  # related to backbone encoder
        self.analytic_adaptor = None
        self.hidden_dim = cfg.hidden_dim
        self.expansion_layer = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim, bias=False, dtype=torch.float),
            nn.ReLU()
            ).to(cfg.device)
        self.clip_model.eval()

    def encode_images(self, images):
        features = self.clip_model.encode_image(images)
        features /= features.norm(dim=-1, keepdim=True)  # normalization to unit vector
        return features

    def analytic_adaption(self, task_id, cfg, train_loader, R):
        if task_id == 0:
            if cfg.feature_expansion:
                # initialize adaptor layer based on 1st dataset class number
                self.analytic_adaptor = nn.Linear(self.hidden_dim, cfg.current_class_num, bias=False).to(cfg.device)
                auto_cor = torch.zeros(self.hidden_dim, self.hidden_dim).to(cfg.device)
                crs_cor = torch.zeros(self.hidden_dim, cfg.current_class_num).to(cfg.device)
            else:
                self.analytic_adaptor = nn.Linear(self.feature_dim, cfg.current_class_num, bias=False).to(cfg.device)
                auto_cor = torch.zeros(self.feature_dim, self.feature_dim).to(cfg.device)
                crs_cor = torch.zeros(self.feature_dim, cfg.current_class_num).to(cfg.device)

            # first task: initialize R_0
            with torch.no_grad():
                # Added impaths
                for i, (images, target, impaths) in \
                        enumerate(tqdm(train_loader, desc=f'Re-Alignment on task-{task_id + 1}', total=len(train_loader),
                                       unit='batch')):
                    images, target = images.to(cfg.device), target.to(cfg.device)
                    train_features = self.encode_images(images)

                    if cfg.feature_expansion:
                        train_features = self.expansion_layer(train_features)

                    train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

                    auto_cor += torch.t(train_features) @ train_features
                    crs_cor += torch.t(train_features) @ (train_labels_one_hot)

            R = np.asmatrix(auto_cor.cpu().numpy() + cfg.regularization * np.eye(train_features.size(1))).I
            R = torch.tensor(R).float().to(cfg.device)

            Delta = R @ crs_cor
            self.analytic_adaptor.weight = torch.nn.parameter.Parameter(torch.t(1.0 * Delta.float()))
            return R

        else:
            # Recursively solving R_t
            w = self.analytic_adaptor.weight.t()
            if cfg.feature_expansion:
                w = torch.cat([w, torch.zeros(self.hidden_dim, cfg.increment).to(cfg.device)], dim=1)
                self.analytic_adaptor = nn.Linear(self.hidden_dim, cfg.current_class_num, bias=False).to(cfg.device)
            else:
                w = torch.cat([w, torch.zeros(self.feature_dim, cfg.increment).to(cfg.device)], dim=1)
                self.analytic_adaptor = nn.Linear(self.feature_dim, cfg.current_class_num, bias=False).to(cfg.device)

            with torch.no_grad():
                for i, (images, target, impaths) in \
                        enumerate(tqdm(train_loader, desc=f'Re-Alignment on task-{task_id + 1}', total=len(train_loader),
                                    unit='batch')):
                    target += cfg.trained_class_num
                    images, target = images.to(cfg.device), target.to(cfg.device)
                    train_features = self.encode_images(images)

                    if cfg.feature_expansion:
                        train_features = self.expansion_layer(train_features)

                    train_labels_one_hot = F.one_hot(target, cfg.current_class_num).float()

                    R = R - R @ train_features.t() @ torch.pinverse(torch.eye(images.size(0)).to(cfg.device) +
                                                                    train_features @ R @ train_features.t()) @ train_features @ R
                    w = w + R @ train_features.t() @ (train_labels_one_hot - train_features @ w)

            self.analytic_adaptor.weight = torch.nn.parameter.Parameter(torch.t(w.float()))
            return R

    def forward(self, images):
        features = self.encode_images(images)
        if cfg.feature_expansion:
            features = self.expansion_layer(features)
        outputs = self.analytic_adaptor(features)
        return outputs

    def zero_shot(self, images, clip_weights):
        features = self.encode_images(images)
        clip_logits = 100. * features @ clip_weights

        return clip_logits


def test_acc(test_loader, cfg):
    top1, top3, top5, test_num = 0.0, 0.0, 0.0, 0.0
    for inputs, targets in tqdm(test_loader, desc='Evaluating on current dataset',
                                total=len(test_loader), unit='batch'):
        targets += cfg.previous_class_num
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        with torch.no_grad():
            outputs = continual_clip_adaptor(inputs)
        acc1, acc3, acc5 = cls_acc(outputs, targets, topk=(1, 3, 5))
        # score_3, score_5 = get_scores(outputs, targets, topk = (3, 5), classnames = )
        top1 += acc1
        top3 += acc3
        top5 += acc5
        test_num += inputs.size(0)

    top1, top3, top5 = (top1 / test_num) * 100, (top3 / test_num) * 100, (top5 / test_num) * 100
    return top1, top3, top5
# def get_scores
import pandas as pd

# Add this code before the test loop (around line 390, after loading the model)
# Load the CSV and extract filenames into a set for efficient lookup
# semantics_df = pd.read_csv('/HDD/Suvam_Backup/Suvam/LADA/semantics.csv')
semantics_df = pd.read_csv('/HDD/Suvam_Backup/Suvam/DIKI/semantics_DIKI.csv')
target_filenames = set(semantics_df['filename'].tolist())


cfg_file = "configs/analytic_clip.yaml"
cfg = yaml.load(open(cfg_file, 'r'), Loader=yaml.Loader)
cfg = EasyDict(cfg)

seed = cfg.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
cfg.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_sequence = cfg.datasets
print("Multi-task dataset sequence: ", dataset_sequence)

# Results
fusion_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
adapter_acc_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))

fusion_top3_acc_table  = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top5_acc_table  = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top3_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_top5_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
fusion_MRR_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))  
fusion_cos_score_table = np.zeros((len(dataset_sequence), len(dataset_sequence)))
in_domain_acc_list = []

cfg.previous_class_num = 0
current_class_names = []
R = None

merged_classnames = []

for _, train_dataset in enumerate(dataset_sequence):
    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'CIFAR100'))
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=-1, preprocess=None, val_transform=None, batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'MNIST'))
    else:
        dataset = build_dataset(train_dataset, DIR_PATH, cfg.num_shots)
    merged_classnames += dataset.classnames
    print(len(dataset.classnames))

"""
Loading model
"""
print('Loading pretrained CLIP model...')

continual_clip_adaptor = continual_clip_adaptor(cfg)
continual_clip_adaptor.clip_model.eval()

train_transform = continual_clip_adaptor.train_preprocess
val_preprocess = continual_clip_adaptor.val_preprocess

"""
Training on dataset sequence
"""
for task_id, train_dataset in enumerate(dataset_sequence):
    print(f"------------------ Start training on task-{task_id + 1}: dataset-{train_dataset}. ---------------------")
    if train_dataset == "cifar100":
        dataset = CIFAR100(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                           batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'CIFAR100'))
    elif train_dataset == "mnist":
        dataset = MNIST(num_shots=cfg.num_shots, preprocess=train_transform, val_transform=val_preprocess,
                        batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'MNIST'))
    else:
        dataset = build_dataset(train_dataset, DIR_PATH, cfg.num_shots)

    current_class_names += dataset.classnames
    cfg.increment = len(dataset.classnames)
    cfg.current_class_num = len(current_class_names)

    if train_dataset == "cifar100" or train_dataset == "mnist":
        train_loader = dataset.train_loader
    else:
        train_loader = build_data_loader(data_source=dataset.train_x, batch_size=cfg.batch_size, tfm=train_transform,
                                         is_train=True, shuffle=True, augmentation_time=cfg.augmentation_time)
    R = continual_clip_adaptor.analytic_adaption(task_id, cfg, train_loader, R)

    cfg.trained_class_num = cfg.current_class_num


    ######## Code to save randomly selected images ########
    
    # generate_human_eval_set_from_sequence(
    # dataset_sequence=dataset_sequence,
    # continual_clip_adaptor=continual_clip_adaptor,
    # merged_classnames=merged_classnames,
    # rankings_path="/HDD/Suvam_Backup/Suvam/rankings.json",
    # device=cfg.device,
    # save_dir="/HDD/Suvam_Backup/Suvam/Regression-based-Analytic-Incremental-Learning/test_images_human_eval",
    # batch_size=cfg.batch_size,
    # val_preprocess=val_preprocess,
    # data_root=DIR_PATH,
    # images_per_dataset=50,
    # topk=3
    # )   

    # breakpoint()
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
            test_set = CIFAR100(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'CIFAR100'))
        elif test_dataset == "mnist":
            test_set = MNIST(num_shots=-1, preprocess=None, val_transform=val_preprocess, batch_size=cfg.batch_size, data_root=os.path.join(DIR_PATH, 'MNIST'))
        else:
            test_set = build_dataset(test_dataset, DIR_PATH, cfg.num_shots)

        if test_dataset == "cifar100" or test_dataset == "mnist":
            test_loader = test_set.test_loader
        else:
            test_loader = build_data_loader(data_source=test_set.test, batch_size=1, is_train=False,
                                            tfm=val_preprocess, shuffle=False)

        template = ['a photo of a {}.']
        # template = ['a photo of a {}.', 'a photo of an {}.']
        clip_weights = clip_classifier(merged_classnames, template, continual_clip_adaptor.clip_model, device=cfg.device)

        class_range_min, class_range_max = tested_cls_num, tested_cls_num + len(test_set.classnames)
        in_domain, in_domain_acc = 0.0, 0.0
        adapter_in_domain, adapter_in_domain_acc = 0.0, 0.0

        top1, top3, top5, test_num = 0.0, 0.0, 0.0, 0.0
        fusion_top1, fusion_top3, fusion_top5 = 0.0, 0.0, 0.0
        top3_score, top5_score = 0.0, 0.0
        fusion_top3_score, fusion_top5_score = 0.0, 0.0
        top_MRR_score = 0.0
        fusion_MRR_score = 0.0
        top_cos_score = 0.0
        fusion_cos_score = 0.0

        for inputs, targets, impaths in tqdm(test_loader, desc=f'Evaluating on dataset-{test_id + 1}: {test_dataset}', # Added impaths
                                    total=len(test_loader), unit='batch'):
            # if any(path in target_filenames for path in impaths) and test_dataset in ["Aircraft"]:
            # if "/HDD/Suvam_Backup/Suvam/DATA/Caltech101/101_ObjectCategories/airplanes/image_0009.jpg" in impaths or "/HDD/Suvam_Backup/Suvam/DATA/Caltech101/101_ObjectCategories/airplanes/image_0069.jpg" in impaths:
                # print("Found target image in Caltech101!")
                # print(impaths)
                # breakpoint()
            test_num += inputs.size(0)

            inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
            targets += tested_cls_num

            with torch.no_grad():
                outputs = continual_clip_adaptor.zero_shot(inputs, clip_weights)  # (B, C_all)
                outputs = F.softmax(outputs, dim=-1)
            predict_cls = torch.argmax(outputs, dim=-1)

            pred = outputs.topk(3)[1]
            print([merged_classnames[i] for i in pred[0]])
            print(impaths)
            ### Following was added to get specific wrong prediction for paper figures
            # else:
            #     continue
            # if "Caltech101" in impaths[0]:
                # breakpoint()
                # Load the top1_pred from semantics.csv for this image
            #     img_filename = os.path.basename(impaths[0])
            #     row = semantics_df[semantics_df['filename'] == img_filename]
            #     if not row.empty:
            #         top1_pred = row.iloc[0]['top1_pred']
            #         pred_class = merged_classnames[pred[0][0].item()]
            #         if str(top1_pred) != str(pred_class):
            #             breakpoint()
            # else:
            #     exit()
            
            # pred = outputs.topk(max(3), 1, True, True)[1].t()
            # breakpoint()
            # Zero-shot acc
            acc1, acc3, acc5 = cls_acc(outputs, targets, topk=(1, 3, 5))
            score_3, score_5 = ranked_acc(outputs, targets, topk=(3, 5), classnames=merged_classnames)    
            top1 += acc1
            top3 += acc3
            top5 += acc5
            top3_score += score_3
            top5_score += score_5
            mrr_score = MRR_Acc(outputs, targets, merged_classnames)
            # breakpoint()
            top_MRR_score += mrr_score
            top_cos_score += Cosine_Score(outputs, targets, merged_classnames)


            # Select ID samples belonging to learned domains by zero-shot
            mask = predict_cls < cfg.current_class_num
            if torch.sum(mask) > 0:
                samples_to_adapt = inputs[mask]
                with torch.no_grad():
                    outputs_adapted = continual_clip_adaptor(samples_to_adapt)

                padding_right = outputs.size(-1) - outputs_adapted.size(-1)
                outputs_adapted = F.pad(outputs_adapted, pad=(0, padding_right, 0, 0), mode='constant', value=0)

                outputs[mask] = (1-cfg.fusion_weight) * outputs[mask] + cfg.fusion_weight * outputs_adapted

            # Fusion acc
            fusion_acc1, fusion_acc3, fusion_acc5 = cls_acc(outputs, targets, topk=(1, 3, 5))
            fusion_score_3, fusion_score_5 = ranked_acc(outputs, targets, topk=(3, 5), classnames=merged_classnames)    
            fusion_top1 += fusion_acc1
            fusion_top3 += fusion_acc3
            fusion_top5 += fusion_acc5
            fusion_top3_score += fusion_score_3
            fusion_top5_score += fusion_score_5
            mrr_score = MRR_Acc(outputs, targets, merged_classnames)
            fusion_MRR_score += mrr_score
            fusion_cos_score += Cosine_Score(outputs, targets, merged_classnames)
            # fusion_cos_score += fusion_cos_score
            

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
        print(f"***** Fusion top-1 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_acc} *****")
        print(f"***** Fusion top-3 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_top3_acc} *****")
        print(f"***** Fusion top-5 acc for dataset-{test_id + 1}: {test_dataset}: {fusion_top5_acc} *****")
        print(f"***** Fusion top-3 custom score for dataset-{test_id + 1}: {test_dataset}: {fusion_top3_score_avg} *****")
        print(f"***** Fusion top-5 custom score for dataset-{test_id + 1}: {test_dataset}: {fusion_top5_score_avg} *****")
        print(f"***** Fusion MRR score for dataset-{test_id + 1}: {test_dataset}: {fusion_MRR_score_avg} *****")
        print(f"***** Fusion Cosine Similarity score for dataset-{test_id + 1}: {test_dataset}: {fusion_cos_score_avg} *****")
        fusion_acc_table[task_id, test_id] = fusion_acc
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
print('Final Fusion Acc Table:')
print(fusion_acc_table)

print("\n===== Final Evaluation Tables =====")

print("\nFusion Top-3 Accuracy Table:")
print(fusion_top3_acc_table)

print("\nFusion Top-5 Accuracy Table:")
print(fusion_top5_acc_table)

print("\nFusion Top-3 Ranked Score Table:")
print(fusion_top3_score_table)

print("\nFusion Top-5 Ranked Score Table:")
print(fusion_top5_score_table)

# breakpoint()

print("\nFusion MRR Score Table:")
print(fusion_MRR_score_table)

print("\nFusion Cosine Similarity Score Table:")
print(fusion_cos_score_table)