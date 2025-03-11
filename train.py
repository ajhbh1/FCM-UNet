from FCM import UNet
from dataset import He_Loader
from torch import optim
import torch.nn as nn
from metrics import *
import logging

def val(net, best_precision, test_loader):
    net = net.eval()
    with torch.no_grad():
        i = 0
        dice_total = 0
        recall_total = 0
        data_val_path.__len__()
        for val_data in test_loader:
            val_image, val_label = val_data
            outputs = net(val_image.to(device, dtype=torch.float32))
            outputs[outputs >= 0] = 1
            outputs[outputs < 0] = 0
            outputs = torch.squeeze(outputs, dim=0)
            val_label = torch.squeeze(val_label, dim=0)
            dice_total += dice_coef(val_label.to(device), outputs)
            recall_total += get_recall(val_label.to(device), outputs)
            i += 1
        aver_dice = dice_total / i
        aver_recall = recall_total / i
        aver_precision = precision_score(aver_dice, aver_recall)
        print(
            'aver_dice=%f,aver_recall=%f,aver_precision=%f' % (aver_dice, aver_recall, aver_precision))
        logging.info(
            'aver_dice=%f,aver_recall=%f,aver_precision=%f' % (aver_dice, aver_recall, aver_precision))

        return best_precision, aver_dice, aver_recall, aver_precision

def train(net, device, data_train_path, data_val_path, epochs=50, batch_size=8, lr=0.0001):
    best_precision, aver_dice, aver_recall, aver_precision = 0, 0, 0, 0
    loss_list = []
    recall_list = []
    dice_list = []
    precision_list = []
    num_epochs = epochs
    lyyl_dataset = He_Loader(data_train_path)
    lyylval_dataset = He_Loader(data_val_path)
    train_loader = torch.utils.data.DataLoader(dataset=lyyl_dataset,
                                               batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(dataset=lyylval_dataset,
                                              batch_size=batch_size)

    optimizer = optim.Adam(net.parameters(),
                           lr=lr,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=0,
                           amsgrad=False)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(num_epochs):
        net = net.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        epoch_loss = 0
        num_batches = 0
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            print("loss/train:", loss.item())
            loss.backward()
            optimizer.step()
            num_batches += 1
            epoch_loss += loss.item()
        if num_batches != 0:
            avg_epoch_loss = epoch_loss / num_batches
            loss_list.append(avg_epoch_loss)
        else:
            print("Warning: No batches processed in epoch", epoch)
        avg_epoch_loss = epoch_loss / num_batches
        loss_list.append(avg_epoch_loss)
        best_precision, aver_dice, aver_recall, aver_precision = val(net, best_precision, test_loader)
        dice_list.append(aver_dice)
        recall_list.append(aver_recall)
        precision_list.append(aver_precision)
        print("epoch %d loss:%0.3f" % (epoch, avg_epoch_loss))
        if (epoch + 1) % 1 == 0:
            torch.save(net.state_dict(), 'output/params_' + str(epoch + 1) + '.pth')
    return net


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = UNet(in_channels=1,  num_classes=1)
    # print(net)
    net.to(device=device)
    data_train_path = "train"
    data_val_path = "val"
    train(net, device, data_train_path, data_val_path)