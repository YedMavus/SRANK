import csv
import numpy as np
from collections import OrderedDict, defaultdict
import torch
import json
import os
import math
import clip
import torch.nn.functional as F

class Evaluator:
    """Evaluator for classification."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset([0])
        
        # CLIP (zero-shot)
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model.eval()

        # CHANGE: Load rankings if available
        self.rankings = None
        with open("/HDD/Suvam_Backup/Suvam/rankings.json") as f:
            self.rankings = json.load(f)
        self.rank_lookup = {
            gt: {e["class"]: e["rank"] for e in entries}
            for gt, entries in self.rankings.items()
        }
        # Initialize CSV file for semantics
        self.csv_path = os.path.join(os.getcwd(), "semantics.csv")
        self.csv_initialized = False

    @torch.no_grad()
    def _clip_cosine(self, a: str, b: str):
        tokens = clip.tokenize([a, b]).to(next(self.clip_model.parameters()).device)
        feats = self.clip_model.encode_text(tokens)
        feats = F.normalize(feats, dim=1)
        # Cosine similarity in [-1, 1], map to [0, 1]
        return ((feats[0] * feats[1]).sum().item() + 1) / 2
    def reset(self, indices):
        self.indices = indices
        self.indices_tensor = torch.tensor(self.indices)
        
        self._correct = 0
        self._coscorrect = 0
        self._topkcorrect = 0
        self._topktotal = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        self._y_conf = []
        
        self._task_pred = []
        self._task_true = []
        
        # CHANGE: Add reciprocal rank tracking
        self.scores = []
        self.topk_acc = []
        self._reciprocal_ranks = []
        self._coscorrect_list = []
    
    def get_scores(self, mo, gt, classnames):
        """
        mo: model output tensor of shape [B, K] (top-k predicted label indices)
        gt: list of GT class names, length B
        classnames: list mapping label index -> class name
        """
        _, pred_idx = mo.topk(k=self.cfg.score_k,dim=1)
        B, K = pred_idx.size()
        scores = []
        # breakpoint()
        for b in range(B):
            preds = pred_idx[b].tolist()
            gt_name = classnames[gt[b]]

            GT_rank_list = self.rankings[gt_name][:K]
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
                        di = K + 1

                    score += math.exp(-0.3 * di) / math.log2(i + 2)
                    score_w += 1.0 / math.log2(i + 2)
                score = score / score_w
                # sum_di = 0
                # for i, cls in enumerate(pred_classes):
                #     if cls in gt_rank_map:
                #         sum_di += abs(i - gt_rank_map[cls])
                #     else:
                #         sum_di += K + 1
                # score = 1.0 / (2 ** sum_di)
            # breakpoint()
            if score>1.0:
                print("err")
            scores.append(score)

        return torch.tensor(scores)
    def get_topk_acc(self, mo, gt):
        _, pred_idx = mo.topk(k=self.cfg.score_k,dim=1)
        hits = pred_idx.eq(gt.unsqueeze(1)).any(dim=1).float()
        return torch.tensor(hits)

    def process(self, mo, gt,impaths=None, classnames=None):
        # mo: model output [batch, num_classes]
        # gt: ground truth [batch]
        pred = mo.max(1)[1]
        # topkcorrect, topktotal = self.get_topk_acc(mo,gt)
        # self._topkcorrect += topkcorrect
        # self._topktotal += topktotal
        conf = torch.softmax(mo, dim=1).max(1)[0]
        matches = pred.eq(gt).float()
        # if len(classnames)>100:
            # if "Aircraft" in impaths[0] or "Food" in impaths[0]:  # Debugging for Aircraft and Food datasets
                # breakpoint()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]
        ## Added for cosine accuracy
        for g, p in zip(gt.tolist(), pred.tolist()):
            gt_name = classnames[g]
            pred_name = classnames[p]
            sim = self._clip_cosine(gt_name, pred_name)
            self._coscorrect += sim
            self._coscorrect_list.append(sim)

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())
        self._y_conf.extend(conf.data.cpu().numpy().tolist())

        task_truth = (gt.unsqueeze(1).cpu() >= self.indices_tensor).int().sum(dim=1) - 1
        task_pred = (pred.unsqueeze(1).cpu() >= self.indices_tensor).int().sum(dim=1) - 1

        self._task_true.extend(task_truth.data.cpu().numpy().tolist())
        self._task_pred.extend(task_pred.data.cpu().numpy().tolist())
        # breakpoint()
        score = self.get_scores(mo,gt,classnames).tolist()
        self.scores.extend(score) 
        # Get top 3 predictions for CSV logging
        top3_values, top3_indices = mo.topk(k=3, dim=1)
        if len(classnames) > 100:
            # Initialize CSV file with headers if not already done
            if not self.csv_initialized:
                with open(self.csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['filename', 'score', 'ground_truth', 'top1_pred', 'top2_pred', 'top3_pred'])
                self.csv_initialized = True
            
            # Write to CSV for scores between 0 and 1
            with open(self.csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for idx, (s, impath, g) in enumerate(zip(score, impaths, gt.tolist())):
                    if 0 < s < 1:
                        top3_preds = [classnames[top3_indices[idx][i].item()] for i in range(3)]
                        gt_name = classnames[g]
                        writer.writerow([impath, s, gt_name] + top3_preds)
        
        self.topk_acc.extend(self.get_topk_acc(mo,gt).tolist())
        # breakpoint()
        for g, p in zip(gt.tolist(), pred.tolist()):
            gt_name = classnames[g]
            pred_name = classnames[p]
            rank = None
            # for entry in self.rankings[gt_name]:
            #     if entry["class"] == pred_name:
            #         rank = entry["rank"]
            #         break
            rank = self.rank_lookup.get(gt_name, {}).get(pred_name)
            if rank is not None:
                rr = 1.0 / rank**(1.5)
            else:
                rr = 0.0
                print(f"Predicted class '{pred_name}' not found in rankings for ground truth '{gt_name}'. Assigning RR=0.")
            self._reciprocal_ranks.append(rr)     


    def evaluate(self):
        indices = self.indices
        results = OrderedDict()

        # Task selection metrics
        self._task_selection_recall = defaultdict(list)
        self._task_selection_precision = defaultdict(list)

        for label, pred in zip(self._task_true, self._task_pred):
            matches = int(label == pred)
            self._task_selection_recall[label].append(matches)
            self._task_selection_precision[pred].append(matches)

        task_id = list(self._task_selection_recall.keys())
        task_id.sort()

        task_selection_recall = []
        task_selection_precision = []
        for id in task_id:
            res = self._task_selection_recall[id]
            correct = sum(res)
            total = len(res)
            task_selection_recall.append(correct / total * 100)
            res = self._task_selection_precision[id]
            correct = sum(res)
            total = len(res)
            task_selection_precision.append(correct / total * 100)

        for id in task_id:
            print(f"* Task {id} selection Recall: {task_selection_recall[id]:.1f}% | Precision: {task_selection_precision[id]:.1f}%")
        print(f"* Average Task selection Recall: {np.mean(task_selection_recall):.1f}% | Precision: {np.mean(task_selection_precision):.1f}%\n")

        # Per-class accuracy
        self._per_class_res = defaultdict(list)

        for label, pred in zip(self._y_true, self._y_pred):
            matches = int(label == pred)
            self._per_class_res[label].append(matches)

        labels = list(self._per_class_res.keys())
        labels.sort()
        
        cls_correct = []
        cls_total = []
        cls_accs = []
        for label in labels:
            res = self._per_class_res[label]
            correct = sum(res)
            cls_correct.append(correct)
            total = len(res)
            cls_total.append(total)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        
        cls_correct = np.array(cls_correct)
        cls_total = np.array(cls_total)
        acc_list = []

        for i in range(len(indices)):
            if i != len(indices) - 1:
                acc_list.append(np.sum(cls_correct[indices[i]:indices[i+1]]) / np.sum(cls_total[indices[i]:indices[i+1]]) * 100)
            else:
                acc_list.append(np.sum(cls_correct[indices[i]:]) / np.sum(cls_total[indices[i]:]) * 100)

        for i in range(len(acc_list)):
            print(f"* Task {i} Accuracy: {acc_list[i]:.1f}%")
        print(f"* Average Accuracy: {np.mean(acc_list):.1f}%")
        
        ### Added score reporting

        # Per-class scores
        self._per_class_scores = defaultdict(list)
        
        for idx,label in enumerate(self._y_true):
            matches = self.scores[idx]
            self._per_class_scores[label].append(matches)

        labels = list(self._per_class_scores.keys())
        labels.sort()
        cls_correct = []
        cls_total = []
        cls_accs = []
        acc_list = []
        # breakpoint()
        for label in labels:
            res = self._per_class_scores[label]
            correct = sum(res)
            cls_correct.append(correct)
            total = len(res)
            cls_total.append(total)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        cls_correct = np.array(cls_correct)
        cls_total = np.array(cls_total)

        for i in range(len(indices)):
            if i != len(indices) - 1:
                acc_list.append(np.sum(cls_correct[indices[i]:indices[i+1]]) / np.sum(cls_total[indices[i]:indices[i+1]]) * 100)
            else:
                acc_list.append(np.sum(cls_correct[indices[i]:]) / np.sum(cls_total[indices[i]:]) * 100)

        for i in range(len(acc_list)):
            print(f"* Task {i} Custom Score: {acc_list[i]:.1f}%")
        print(f"* Average Custom Score: {np.mean(acc_list):.1f}%")

        ### Added top-K reporting

        # Per-class scores
        self._per_class_topk = defaultdict(list)

        for idx,_ in enumerate(self._y_true):
            matches = self.topk_acc[idx]
            self._per_class_topk[self._y_true[idx]].append(matches)

        labels = list(self._per_class_topk.keys())
        labels.sort()
        cls_correct = []
        cls_total = []
        cls_accs = []
        acc_list = []
        for label in labels:
            res = self._per_class_topk[label]
            correct = sum(res)
            cls_correct.append(correct)
            total = len(res)
            cls_total.append(total)
            acc = 100.0 * correct / total
            cls_accs.append(acc)
        cls_correct = np.array(cls_correct)
        cls_total = np.array(cls_total)

        for i in range(len(indices)):
            if i != len(indices) - 1:
                acc_list.append(np.sum(cls_correct[indices[i]:indices[i+1]]) / np.sum(cls_total[indices[i]:indices[i+1]]) * 100)
            else:
                acc_list.append(np.sum(cls_correct[indices[i]:]) / np.sum(cls_total[indices[i]:]) * 100)

        for i in range(len(acc_list)):
            print(f"* Task {i} Top-K Score: {acc_list[i]:.1f}%")
        print(f"* Average Top-K Score: {np.mean(acc_list):.1f}%")

        ## Added reciprocal rank code
        if len(self._reciprocal_ranks) > 0:
            print("\n" + "="*80)
            print("Reciprocal Rank Metrics")
            print("="*80)
            # breakpoint()
            # Overall MRR
            rr_array = np.array(self._reciprocal_ranks)
            mrr = rr_array.mean()
            print(f"* Mean Reciprocal Rank (MRR): {mrr:.4f}")
            
            # Per-task MRR
            y_true_array = np.array(self._y_true)
            
            task_mrr_list = []
            for i in range(len(indices)):
                if i != len(indices) - 1:
                    task_mask = (y_true_array >= indices[i]) & (y_true_array < indices[i+1])
                else:
                    task_mask = (y_true_array >= indices[i])
                
                if task_mask.sum() > 0:
                    task_mrr = rr_array[task_mask].mean()
                    task_mrr_list.append(task_mrr)
                    print(f"* Task {i} MRR: {task_mrr:.4f}")
            
            if len(task_mrr_list) > 0:
                print(f"* Average Task MRR: {np.mean(task_mrr_list):.4f}")
            
            print("="*80 + "\n")
            
            # Store in results
            results['mrr'] = mrr
            results['task_mrr'] = task_mrr_list if len(task_mrr_list) > 0 else []

        self._per_class_res = defaultdict(list)

        for idx, (label, pred) in enumerate(zip(self._y_true, self._y_pred)):
            sim = self._coscorrect_list[idx]   # cosine similarity
            self._per_class_res[label].append(sim)

        labels = sorted(self._per_class_res.keys())

        cls_sum = []
        cls_total = []

        for label in labels:
            sims = self._per_class_res[label]
            cls_sum.append(np.sum(sims))
            cls_total.append(len(sims))

        cls_sum = np.array(cls_sum)
        cls_total = np.array(cls_total)

        acc_list = []

        for i in range(len(indices)):
            if i != len(indices) - 1:
                task_sum = np.sum(cls_sum[indices[i]:indices[i+1]])
                task_cnt = np.sum(cls_total[indices[i]:indices[i+1]])
            else:
                task_sum = np.sum(cls_sum[indices[i]:])
                task_cnt = np.sum(cls_total[indices[i]:])

            task_acc = 100.0 * task_sum / task_cnt
            acc_list.append(task_acc)

        for i, acc in enumerate(acc_list):
            print(f"* Task {i} Semantic Accuracy (CLIP): {acc:.2f}%")

        print(f"* Average Semantic Accuracy (CLIP): {np.mean(acc_list):.2f}%")

        results['accuracy'] = np.mean(acc_list)
        results['task_accuracies'] = acc_list




        return results