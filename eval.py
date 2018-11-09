import torch
import torch.nn.functional as F
import numpy as np
from dice_loss import dice_coeff
#    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
#            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
# dis =2ab/a
def jaccard_distance(target, prediction):
    intersection = np.logical_and(target, prediction)
    union = np.logical_or(target, prediction)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    total_jaccard = 0
    for i, b in enumerate(dataset):
        #print("i,b:"i,b)
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

        mask_pred = mask_pred.cpu().numpy()
        true_mask = true_mask.cpu().numpy()
        total_jaccard += jaccard_distance(mask_pred, true_mask)
    m = tot / i
    n = total_jaccard / i
    return m, n
