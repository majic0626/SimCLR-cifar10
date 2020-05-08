import torch.optim as optim
import torch
import torch.nn as nn
import json
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import args
from models import Resnet18, Projector, Linear_Classifier
from criterions import NTXEntLoss
from helper import visualize, adjust_lr, adjust_linear_lr, LARC


def Train_CNN():
    model.train()
    train_loss = 0.0
    opt_cnn.zero_grad()
    for ix, (sample1, sample2) in enumerate(trainloader):
        data_i, data_j = sample1["image"], sample2["image"]
        data_i, data_j = data_i.to(device), data_j.to(device)
        h_i = model(data_i)
        h_j = model(data_j)
        z_i = g(h_i)
        z_j = g(h_j)
        loss = criterion1(z_i, z_j)  # NT-Xent Loss in the paper
        loss = loss / accum
        loss.backward()
        train_loss += loss.item()
        if (ix + 1) % accum == 0:
            opt_cnn.step()
            opt_cnn.zero_grad()
        if ((ix + 1) % (5 * accum)) == 0:
            print("L-train loss:{}".format(train_loss * accum / (ix + 1)))

    record_cnn["train_loss"].append(train_loss * accum / (ix + 1))


def Train_CLF():
    model.eval()
    linear_clf.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for ix, (sample1, sample2) in enumerate(Linear_trainloader):
        opt_clf.zero_grad()
        data, label = sample1["image"], sample1["label"]
        data, label = data.to(device), label.to(device)
        feature = model(data)
        output = linear_clf(feature.detach())
        loss = criterion2(output, label)
        loss.backward()
        opt_clf.step()
        train_loss += loss.item()
        _, predict = output.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
        if (ix + 1) % 100 == 0:
            print("L-train loss:{} / L-acc:{}".format(train_loss / (ix + 1),
                                                      100 * correct / total))
    record_clf["train_loss"].append(train_loss / (ix + 1))
    record_clf["train_acc"].append(100 * correct / total)


def Test_CLF():
    model.eval()
    linear_clf.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    global best_acc
    with torch.no_grad():
        for ix, (sample1, sample2) in enumerate(Linear_testloader):
            data, label = sample1["image"], sample1["label"]
            data, label = data.to(device), label.to(device)
            feature = model(data)
            output = linear_clf(feature.detach())
            loss = criterion2(output, label)
            test_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
        print("L-test loss:{} / L-acc:{}".format(test_loss / (ix + 1),
                                                 100 * correct / total))
        record_clf["test_loss"].append(test_loss / (ix + 1))
        record_clf["test_acc"].append(100 * correct / total)
    return 100 * correct / total


def record_saver(record, path):
    with open(path, 'w') as f:
        json.dump(record, f)


if __name__ == "__main__":
    # ========== [param] ==========
    for arg in vars(args):
        print(arg, '===>', getattr(args, arg))
    lr = args.lr
    clf_lr = args.clf_lr
    batch_size = args.batch
    epoch = args.epoch
    clf_epoch = args.clf_epoch
    classNum = args.classNum
    temp = args.temperature
    data_root = args.data_root
    train_root = data_root + '/' + "train"
    test_root = data_root + '/' + "test"
    classFile = data_root + '/' + "class.txt"
    num_worker = args.workers
    dir_ckpt = args.dir_ckpt
    dir_log = args.dir_log
    accum = args.accumulate
    aug_s = args.strength
    useLARS = args.useLARS
    decay = args.weight_decay
    momentnum = args.momentnum
    warm = args.warmup
    project_in = args.pro_in
    project_hidden = args.pro_hidden
    project_out = args.pro_out
    linear_in = args.linear_in
    eval_routine = args.eval_routine
    record_cnn = {"train_loss": []}
    record_clf = {"train_loss": [],
                  "train_acc": [],
                  "test_loss": [],
                  "test_acc": []}

    # ========== [data] ==========
    train_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=32),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.8 * aug_s,
                                                       contrast=0.8 * aug_s,
                                                       saturation=0.8 * aug_s,
                                                       hue=0.2 * aug_s)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor()]
    )

    trainset = ImageDataset(
        root_dir=train_root,
        class_file=classFile,
        transforms=train_aug
    )

    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_worker
    )

    Linear_train_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size=32),
        transforms.ToTensor()]
    )

    Linear_trainset = ImageDataset(
        root_dir=train_root,
        class_file=classFile,
        transforms=Linear_train_aug
    )

    Linear_trainloader = DataLoader(
        Linear_trainset,
        batch_size=256,
        shuffle=True,
        num_workers=num_worker
    )

    Linear_test_aug = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor()]
    )

    Linear_testset = ImageDataset(
        root_dir=test_root,
        class_file=classFile,
        transforms=Linear_test_aug
    )

    Linear_testloader = DataLoader(
        Linear_testset,
        batch_size=256,
        shuffle=False,
        num_workers=num_worker
    )

    # ========== [visualize] ==========
    if batch_size >= 64:
        visualize(trainloader, dir_log + '/' + 'visual.png')

    # ========== [device] =============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== [cnn model] ==========
    model = Resnet18()
    model.to(device)
    g = Projector(input_size=project_in, hidden_size=project_hidden, output_size=project_out)
    g.to(device)

    # ========== [optim for cnn] ==========
    opt_cnn = optim.SGD(
        list(model.parameters()) + list(g.parameters()),
        lr=lr,
        momentum=momentnum,
        weight_decay=decay
    )

    criterion1 = NTXEntLoss(temp=temp)
    criterion2 = nn.CrossEntropyLoss()

    if useLARS:
        opt_cnn = LARC(opt_cnn)  # LARS on SGD optimizer

    best_acc = 0.0
    for i in range(1, epoch + 1):
        print("========== [Unsupervised Training] ==========")
        print("[epoch {}/{}]".format(i, epoch))
        print("[lr {}]".format(adjust_lr(opt=opt_cnn, epoch=i, lr_init=lr, T=epoch, warmup=warm)))
        Train_CNN()
        record_saver(record_cnn, dir_log + '/' + "cnn.txt")
        if (i % eval_routine) == 0:
            linear_clf = Linear_Classifier(input_size=linear_in, classNum=classNum)
            linear_clf.to(device)
            opt_clf = optim.SGD(linear_clf.parameters(),
                                lr=clf_lr,
                                momentum=momentnum,
                                )
            if useLARS:
                opt_clf = LARC(opt_clf)
            for j in range(1, clf_epoch + 1):
                print("========== [Supervised Training] ==========")
                print("[epoch {}/{}]".format(j, clf_epoch))
                print("[lr {}]".format(adjust_linear_lr(opt=opt_clf, epoch=j, lr_init=clf_lr, T=clf_epoch)))
                Train_CLF()
                print("========== [Supervised Testing] ==========")
                test_acc = Test_CLF()
                print("save the last model: {} || best model: {}".format(test_acc, best_acc))
                torch.save({"cnn": model.state_dict(), "clf": linear_clf.state_dict(), "epoch": j}, dir_ckpt + '/' + "last.pt")
                record_saver(record_clf, dir_log + '/' + "clf.txt")
                if test_acc > best_acc:
                    best_acc = test_acc
                    print("save the best model: {}".format(best_acc))
                    torch.save({"cnn": model.state_dict(), "clf": linear_clf.state_dict(), "epoch": j}, dir_ckpt + '/' + "best.pt")
