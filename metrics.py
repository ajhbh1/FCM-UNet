import torch

def dice_coef(output, target):
    smooth = 1e-5
    output=torch.sigmoid(output)
    output[output > 0.5] = 1
    output[output <= 0.5] = 0
    intersection = (output * target).sum()
    dice = (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)
    return dice

def get_recall(output, predict):
    output = output > 0.5
    predict = predict > 0.5
    TP = ((output==1).byte() + (predict==1).byte()) == 2
    FN = ((output==0).byte() + (predict==1).byte()) == 2
    SE = float(torch.sum(TP)) / (float(torch.sum(TP + FN)) + 1e-6)
    return SE

def precision_score(dice,recall):
    percision = recall * dice.item() / (2 * recall - dice.item())
    return percision



