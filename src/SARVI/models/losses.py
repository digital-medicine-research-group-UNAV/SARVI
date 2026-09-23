import math
import torch
import torch.nn as nn

class MoMLoss(nn.Module):
    def __init__(self, M, majority_label, delta, sentence_loss_fn, majority_loss_fn, ignore_index=-100):
        super().__init__()
        self.M = M
        self.majority_label = majority_label
        self.delta = float(delta)
        self.sentence_loss_fn = sentence_loss_fn
        self.majority_loss_fn = majority_loss_fn
        self.ignore_index = ignore_index

        # must be reduction='none'
        if sentence_loss_fn.reduction != "none":
            raise ValueError("sentence_loss_fn must use reduction='none'")
        if majority_loss_fn.reduction != "none":
            raise ValueError("majority_loss_fn must use reduction='none'")

    def forward(self, logits, targets, weights=None):
        """
        logits: (B, T, C) for BIO or (N, C) for Span classification
        targets: (B, T) for BIO or (N,) for Span classification
        weights: optional, same shape as targets
        """
        if logits.dim() == 2:
            logits = logits.unsqueeze(0)
            targets = targets.unsqueeze(0)
            if weights is not None:
                weights = weights.unsqueeze(0)

        B, T, C = logits.shape

        if weights is not None:
            weights = weights.to(device=logits.device, dtype=logits.dtype)

        # flatten for CE
        flat_logits = logits.reshape(-1, C)
        flat_targets = targets.reshape(-1)

        # per-token losses
        full_token_loss = self.sentence_loss_fn(flat_logits, flat_targets)
        maj_token_loss = self.majority_loss_fn(flat_logits, flat_targets)

        full_token_loss = full_token_loss.view(B, T)
        if weights is not None:
            full_token_loss = full_token_loss * weights
        maj_token_loss = maj_token_loss.view(B, T)

        valid_mask = targets != self.ignore_index
        valid_counts = valid_mask.sum(dim=1).clamp(min=1)

        # ----- L(y(n), p(n)) -----
        L = (full_token_loss * valid_mask).sum(dim=1) / valid_counts  # mean over tokens

        # ----- LMoM(y(n), p(n)) -----
        majority_mask = (targets == self.majority_label) & valid_mask
        LMoM = (maj_token_loss * majority_mask).sum(dim=1) / self.M

        # ----- Lsentence -----
        Lsentence = self.delta * L + (1.0 - self.delta) * LMoM

        # ----- Lmodel -----
        return Lsentence.mean()

### Copied directly from: https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class CombinedMarginLoss(nn.Module):
    def __init__(self, 
                 s, 
                 m1,
                 m2,
                 m3,
                 interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold
        
        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False


    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty    
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s        

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits

class ArcFace(torch.nn.Module):
    """ ArcFace (https://arxiv.org/pdf/1801.07698v1.pdf):
    """
    def __init__(self, s=64.0, margin=0.5):
        super(ArcFace, self).__init__()
        self.s = s
        self.margin = margin
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.theta = math.cos(math.pi - margin)
        self.sinmm = math.sin(math.pi - margin) * margin
        self.easy_margin = False


    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]

        with torch.no_grad():
            target_logit.arccos_()
            logits.arccos_()
            final_target_logit = target_logit + self.margin
            logits[index, labels[index].view(-1)] = final_target_logit
            logits.cos_()
        logits = logits * self.s   
        return logits


class CosFace(torch.nn.Module):
    def __init__(self, s=64.0, m=0.40):
        super(CosFace, self).__init__()
        self.s = s
        self.m = m

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        index = torch.where(labels != -1)[0]
        target_logit = logits[index, labels[index].view(-1)]
        final_target_logit = target_logit - self.m
        logits[index, labels[index].view(-1)] = final_target_logit
        logits = logits * self.s
        return logits