import os
import numpy as np
import random
import torch
import math
import torch.nn.functional as F
import json

from clip import clip


def cosine_schedule_warmup(total_step, value, final_value=0, warmup_step=0, warmup_value=0):
    if warmup_step > 0:
        warmup_schedule = np.linspace(warmup_value, value, warmup_step+2)[1:-1]
    else:
        warmup_schedule = np.array([])
    steps = np.arange(total_step - warmup_step)
    schedule = final_value + 0.5 * (value-final_value) * (1+np.cos(np.pi * steps / len(steps)))
    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_step
    return schedule

class build_cosine_scheduler:
    def __init__(self, optimizer, lr, total_step, lr_warmup_step=0):
        init_lr = 0
        final_lr = lr * 1e-3
        self.lrs = cosine_schedule_warmup(total_step, lr, final_lr, lr_warmup_step, init_lr)
        self.optimizer = optimizer

    def step(self,idx):
        lr = self.lrs[idx]
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"]= lr
        self.lr=lr


def get_transform(cfg):
    return clip._transform(cfg.input_size[0])


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_loss_3d(pred, target):
    pred = pred / pred.norm(dim=-1, keepdim=True)
    target = target / target.norm(dim=-1, keepdim=True)
    loss = torch.sum(pred*target, dim=2)
    loss = 1 - torch.mean(loss)
    return loss

def cal_MTIL_metrics(acc_list):
    acc_list = np.array(acc_list)
    acc_list *= 100
    avg = acc_list.mean(axis=0)
    last = np.array(acc_list[-1, :])
    transfer = np.array([np.mean([acc_list[j, i] for j in range(i)]) for i in range(1, acc_list.shape[1])])
    g = lambda x: np.around(x.mean(), decimals=1) if len(x) > 0 else -1
    f = lambda x: [np.around(i, decimals=1) for i in x]
    return {"transfer": {"transfer": f(transfer)}, "avg": {"avg": f(avg)}, "last": {"last": f(last)}, 
            "results_mean": {"transfer": g(transfer), "avg": g(avg), "last": g(last)}}



def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]

import re

def canonicalize_classname(name: str) -> str:
    """
    Converts:
    'a photo of a 747-200, a type of aircraft.'
    -> '747-200'
    """
    
    name = name.lower()

    # common CLIP template pattern
    name = name.replace("a photo of", "")
    name = name.replace("a type of aircraft", "")
    name = name.replace(",", " ")

    name = name.strip()

    # extract aircraft token (letters + numbers + hyphens)
    match = re.search(r"[a-z]*\d+(?:-\d+)?", name)
    if match:
        return match.group(0).upper()
    else:
        breakpoint()
    raise ValueError(f"Could not canonicalize classname: {name}")

# # Read the ranks
rankings = None
with open("/HDD/Suvam_Backup/Suvam/rankings.json") as f: rankings = json.load(f)

def ranked_acc(output, target, classnames, rankings= rankings, topk=(3,5)):
    """
    output: Tensor [B, C] (logits)
    target: Tensor [B] (GT label indices)
    classnames: list[str], idx -> class name
    rankings: dict loaded from rankings.json
    topk: tuple of k values
    """
    
    maxk = max(topk)
    _, pred_idx = output.topk(maxk, dim=1, largest=True, sorted=True)
    # canonical_names = [canonicalize_classname(name) for name in classnames]
    canonical_names = classnames
    # breakpoint()
    B = output.size(0)
    scores_per_k = []

    for k in topk:
        scores = []

        for b in range(B):
            preds = pred_idx[b, :k].tolist()
            gt_name = canonical_names[target[b].item()]

            GT_rank_list = rankings[gt_name][:k]
            gt_rank_map = {
                item["class"]: item["rank"] - 1
                for item in GT_rank_list
            }

            pred_classes = [canonical_names[i] for i in preds]

            if GT_rank_list[0]["class"] == pred_classes[0]:
                score = 1.0
            elif GT_rank_list[0]["class"] not in pred_classes:
                score = 0.0
            else:
                score = 0.0
                score_w = 0.0
                for i, cls in enumerate(pred_classes):
                    if cls in gt_rank_map:
                        di = abs(i - gt_rank_map[cls])
                    else:
                        di = k + 1

                    w = 1.0 / math.log2(i + 2)
                    score += math.exp(-0.3 * di) * w
                    score_w += w

                score = score / score_w

            scores.append(score)

        # cls_acc returns summed corrects; keep same convention
        scores_per_k.append(float(torch.tensor(scores).sum().item()))

    return scores_per_k
def MRR_Acc(output, target, classnames, rankings=rankings):
    top1_idx = output.argmax(dim=1)  # [B]
    B = output.size(0)
    # canonical_names = [canonicalize_classname(name) for name in classnames]
    canonical_names = classnames
    total_score = 0.0

    for b in range(B):
        gt_name = canonical_names[target[b].item()]
        pred_name = canonical_names[top1_idx[b].item()]

        rr = 0.0
        for item in rankings[gt_name]:
            if item["class"] == pred_name:
                rr = 1.0 / item["rank"]  # rank is 1-based
                break

        total_score += rr

    return float(total_score)

def load_vanilla_clip(device):
    model_name = "ViT-B/16"
    url = clip._MODELS[model_name]
    model_path = clip._download(url)

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    design_details = {
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
        "pool_size": 1,
    }

    model = clip.build_model(
        state_dict or model.state_dict(),
        design_details
    ).to(device)

    preprocess = clip._transform(model.visual.input_resolution)
    return model, preprocess

device ="cuda" if torch.cuda.is_available() else "cpu"
# cfg.device must already exist
clip_model, val_preprocess = load_vanilla_clip(device)
clip_model.eval()

TEXT_EMB = None

@torch.no_grad()
def build_text_embedding_cache(classnames):
    global TEXT_EMB
    
    TEXT_EMB = {}
    tokens = clip.tokenize(classnames).to(device)
    emb = clip_model.encode_text(tokens)
    emb = F.normalize(emb, dim=1)

    for i, name in enumerate(classnames):
        TEXT_EMB[name] = emb[i]
@torch.no_grad()
def Cosine_Score(output, target, classnames):
    """
    output: Tensor [B, C]
    target: Tensor [B]
    classnames: list[str]

    Returns:
        summed cosine similarity over batch
    """
    # canonical_names = [canonicalize_classname(name) for name in classnames]
    canonical_names = classnames
    build_text_embedding_cache(canonical_names)
    # assert TEXT_EMB is not None, "Call build_text_embedding_cache() first"

    top1_idx = output.argmax(dim=1)
    B = output.size(0)

    score = 0.0

    for b in range(B):
        pred_name = canonical_names[top1_idx[b].item()]
        gt_name   = canonical_names[target[b].item()]

        score += (torch.dot(TEXT_EMB[pred_name],TEXT_EMB[gt_name]).item()+1)/2
        # breakpoint()

    return float(score)

