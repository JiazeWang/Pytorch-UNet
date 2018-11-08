import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff


def jaccard_distance(y_true, y_pred):
    intersection = torch.sum(torch.abs(y_true * y_pred), dim=-1)
    sum_ = torch.sum(torch.abs(y_true) + torch.abs(y_pred), dim=-1)
    jac = (intersection) / (sum_ - intersection)
    return jac

def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    total_jaccard = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)[0]
        mask_pred = (F.sigmoid(mask_pred) > 0.5).float()

        tot += dice_coeff(mask_pred, true_mask).item()
        total_jaccard += jaccard_distance(mask_pred, true_mask)
    return tot / i, total_jaccard / i
