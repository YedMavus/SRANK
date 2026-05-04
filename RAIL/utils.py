import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np
import math
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


class kernel_layer(nn.Module):
    def __init__(self, sv, gamma):
        super(kernel_layer, self).__init__()
        self.sv = sv
        self.gamma = gamma

    def forward(self, x):
        return kernel(x, self.sv, gamma=self.gamma)


def cls_acc(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        for k in topk
    ]



# # Read the ranks
ranking_path = 'path to ranks'
rankings = None
with open(ranking_path) as f: rankings = json.load(f)

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

    B = output.size(0)
    scores_per_k = []

    for k in topk:
        scores = []

        for b in range(B):
            preds = pred_idx[b, :k].tolist()
            gt_name = classnames[target[b].item()]

            GT_rank_list = rankings[gt_name][:k]
            gt_rank_map = {
                item["class"]: item["rank"] - 1
                for item in GT_rank_list
            }

            pred_classes = [classnames[i] for i in preds]

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

    total_score = 0.0

    for b in range(B):
        gt_name = classnames[target[b].item()]
        pred_name = classnames[top1_idx[b].item()]

        rr = 0.0
        for item in rankings[gt_name]:
            if item["class"] == pred_name:
                rr = 1.0 / item["rank"]  # rank is 1-based
                break

        total_score += rr

    return float(total_score)
device ="cuda" if torch.cuda.is_available() else "cpu"
# cfg.device must already exist
clip_model, train_preprocess, val_preprocess = clip.load(
    "ViT-B/16",
    device=device,
    jit=False
)

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
    build_text_embedding_cache(classnames)
    # assert TEXT_EMB is not None, "Call build_text_embedding_cache() first"

    top1_idx = output.argmax(dim=1)
    B = output.size(0)

    score = 0.0

    for b in range(B):
        pred_name = classnames[top1_idx[b].item()]
        gt_name   = classnames[target[b].item()]

        score += (torch.dot(TEXT_EMB[pred_name],TEXT_EMB[gt_name]).item()+1)/2
        # breakpoint()

    return float(score)



def one_hot_cls_acc(output, target):
    if isinstance(output, np.ndarray) and isinstance(target, np.ndarray):
        pred = np.argmax(output, axis=1)
        labels = np.argmax(target, axis=1)
        correct_predictions = np.equal(pred, labels)
        acc = np.mean(correct_predictions.astype(float)) * 100
    elif torch.is_tensor(output) and torch.is_tensor(target):
        pred = torch.argmax(output, dim=1)
        labels = torch.argmax(target, dim=1)
        correct_predictions = torch.eq(pred, labels)
        acc = torch.mean(correct_predictions.float()) * 100
    else:
        raise ValueError('Unsupported types for prediction and target.')
    return acc


def clip_classifier(classnames, template, clip_model, device="cuda"):
    with torch.no_grad():
        clip_weights = []

        for classname in classnames:
            # Tokenize prompts
            classname = classname.replace('_', ' ')
            texts = [t.format(classname) for t in template]
            texts = clip.tokenize(texts).to(device)

            class_embeddings = clip_model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)

            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            clip_weights.append(class_embedding)

        clip_weights = torch.stack(clip_weights, dim=1).to(device)
    return clip_weights


def encode_images(clip_model, images):
    """
        forward pass of CLIP image encoder to extract unit vector features
    """
    features = clip_model.encode_image(images)
    features /= features.norm(dim=-1, keepdim=True)


    return features

def kernel(x, X, gamma):
    """
    Args:
        x: input data
        X: static center embeddings
        gamma: Guassian kernel hyperparameter
    """
    with torch.no_grad():
        btch = 32
        ker = torch.exp(((X[:btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
        for i in range(1, math.ceil(X.size(0) / btch)):
            ker_new = torch.exp(
                ((X[i * btch:(i + 1) * btch, :] - x.unsqueeze(1)) ** 2).sum(dim=-1).mul_(-1. * gamma))
            ker = torch.cat((ker, ker_new), 1)
    return ker


def gaussian_kernel(x, X, gamma):
    distance = pairwise_distances(x, X, metric='euclidean', squared=True)
    return np.exp(-gamma * distance)


def linear_kernel(x, X):
    return x @ X.T


def cos_kernel(x, X):
    return 1 - linear_kernel(x, X)


def sample_per_class(dataset, n, num_classes=1000):
    indices_per_class = [[] for _ in range(num_classes)]
    for idx, (_, label) in enumerate(dataset.imgs):
        indices_per_class[label].append(idx)

    sampled_indices = [idx for indices in indices_per_class for idx in np.random.choice(indices, n, replace=False)]
    return sampled_indices


class kernel_ridge_regression:
    def __init__(self, lamda=0.1, gamma=0.1):
        self.lamda = lamda
        self.gamma = gamma
        self.alpha = None
        self.kernel = None

    def train(self, X, Y):
        """
        Gaussian kernel only
        """
        self.kernel = kernel(X, X, gamma=self.gamma).cpu().numpy()
        self.alpha = np.asmatrix(self.kernel + self.lamda * np.eye(self.kernel.shape[0])).I @ Y
        return self.alpha

    def predict(self, X, X_train):
        """
        Args:
            X: on-device tensor
            X_train: on-device tensor
        Returns:
            on-cpu numpy
        """
        predictions = kernel(X, X_train, gamma=self.gamma).cpu().numpy() @ self.alpha
        return predictions
