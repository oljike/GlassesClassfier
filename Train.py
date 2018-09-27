from torch.utils.data import DataLoader
from DataUtils import GlassDataset
import torch.optim as optim
import torch
from Models import OlzhasNet45, ResNet18, MobileNetV2
from torch.autograd import Variable
import torch.nn as nn
from sklearn import metrics
import datetime
import numpy as np
from torchvision import transforms
import argparse


def compute_accuracy(prob_cls, gt_cls, threshold=0.5):
    prob_cls = torch.squeeze(prob_cls)
    gt_cls = torch.squeeze(gt_cls)

    size = min(gt_cls.size()[0], prob_cls.size()[0])
    prob_ones = torch.ge(prob_cls, threshold).float()

    right_ones = torch.eq(prob_ones, gt_cls).float()

    return torch.div(torch.mul(torch.sum(right_ones), float(1.0)), float(size))


def train(model_name, lr=0.05, pretrained=None):

    train_dataset = GlassDataset('annotations/anno_train.txt',
                                 transform=transforms.Compose([
                                     transforms.ToTensor()
                                 ]))
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=4

                              )

    test_dataset = GlassDataset('annotations/anno_test.txt',
                                transform=transforms.Compose([
                                    transforms.ToTensor()
                                ])
                                )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=4)

    if model_name == 'ResNet':
        net = ResNet18()
    elif model_name == 'MobileNet':
        net = MobileNetV2()
    elif model_name == 'OlzhasNet':
        net = OlzhasNet45()
    else:
        print("No such model!")
        quit()

    if pretrained is not None:
        state_dict = torch.load(pretrained)
        net.load_state_dict(state_dict)

    net.train()
    net.cuda()

    lossfn = nn.BCELoss().cuda()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    for cur_epoch in range(1, 50):
        train_preds = []
        train_gts = []
        acc_train = []
        loss_train = 0

        for batch_idx, data in enumerate(train_loader, 0):

            img_batch, label_batch = data['img'], data['label']

            im_tensor = img_batch.float().cuda()
            gt_label = label_batch.float().cuda()

            cls_pred = net(im_tensor)

            cls_pred = cls_pred.squeeze(1)

            cls_loss = lossfn(cls_pred, gt_label)


            cls_pred = cls_pred.cpu().detach()
            gt_label = gt_label.cpu().detach()

            train_preds.append(cls_pred)
            train_gts.append(gt_label)
            loss_train += cls_loss.data.item()

            if batch_idx % 10 == 0:

                train_preds = torch.cat(train_preds, dim=0)
                train_gts = torch.cat(train_gts, dim=0)

                accuracy = compute_accuracy(train_preds, train_gts)
                acc_train.append(accuracy)

                print("%s : Epoch: %d, Step: %d, accuracy: %s, label loss: %s, lr:%s "
                      % (datetime.datetime.now(), cur_epoch, batch_idx, accuracy, str(loss_train / 200), lr))

                train_preds = []
                train_gts = []

            optimizer.zero_grad()
            cls_loss.backward()
            optimizer.step()

        accuracy_avg = np.mean(acc_train)
        cls_loss_avg = loss_train / batch_idx

        print("Train: epoch: %d, accuracy: %s, cls loss: %s" % (cur_epoch, accuracy_avg, cls_loss_avg))

        accuracy_test, recall_test, precision_test = eval(test_loader, net, cur_epoch, model_name)
        f1 = 2 * (precision_test * recall_test) / (precision_test + recall_test)
        print(
            "Test: accuracy: %s, recall: %s, precision: %s, f1: %s" % (accuracy_test, recall_test, precision_test, f1))

    print('Finished Training')


def eval(test_loader, net, epoch, model_name):
    net.eval()
    test_preds = []
    test_gts = []

    for batch_idx, test_data in enumerate(test_loader, 0):
        # get the inputs
        test_img, test_label = test_data['img'], test_data['label']

        test_tensor = Variable(test_img).float()
        test_gt = Variable(test_label).float()

        test_tensor = test_tensor.cuda()
        test_gt = test_gt.cuda()

        test_pred = net(test_tensor)


        test_pred = test_pred.cpu().data
        test_gt = test_gt.cpu().data

        test_preds.append(test_pred)
        test_gts.append(test_gt)

    print(test_pred)
    accuracy_test = compute_accuracy(torch.cat(test_preds, dim=0), torch.cat(test_gts, dim=0))
    test_preds = np.concatenate(np.around(torch.cat(test_preds, dim=0).numpy(), decimals=0), axis=0)
    test_gts = np.around(torch.cat(test_gts, dim=0).numpy(), decimals=0)


    recall_test = metrics.recall_score(test_gts, test_preds)
    precision_test = metrics.precision_score(test_gts, test_preds)

    torch.save(net.state_dict(),
               "./weights/" + model_name + "/" + model_name + "_epoch_%d-%.2f-%.2f-%.2f.pt" % (
               epoch, accuracy_test, recall_test, precision_test))

    return accuracy_test, recall_test, precision_test


def parse_args():
    '''Parse command line arguments'''
    parser = argparse.ArgumentParser(formatter_class=
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', default=0.00001,
                        help='Learning Rate')
    parser.add_argument('--model_name', default='MobileNet',
                        help='Possible options: ResNet, MobileNet')
    parser.add_argument('--pretrained',
                        default='/home/oljike/PycharmProjects/GlassesClassification/weights/MobileNet/MobileNet_epoch_8-0.94-0.83-0.84.pt',
                        help='Path to pretrained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args.model_name, lr=args.lr, pretrained = args.pretrained)
