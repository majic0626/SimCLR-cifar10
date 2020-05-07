# test.py
import torch.optim as optim
import torch
import torch.nn as nn
from dataset import ImageDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from config import args
from models import ResNet, Linear_Classifier
from helper import visualize, adjust_linear_lr


def Train_CLF(ep, fine_tune):
    if fine_tune:
        model.train()
    else:
        model.eval()
    linear_clf.train()
    lr_ = adjust_linear_lr(opt=opt_clf, epoch=ep, lr_init=5e-3, lr_end=5e-5, T=100)
    train_loss = 0.0
    correct = 0
    total = 0
    print("========== [Supervised Training] ==========")
    print("[epoch {}]".format(ep))
    print("[lr {}]".format(lr_))
    for ix, (sample1, sample2) in enumerate(Linear_trainloader):
        opt_clf.zero_grad()
        data, label = sample1["image"], sample1["label"]
        data, label = data.to(device), label.to(device)
        feature = model(data)
        if not fine_tune:
            feature = feature.detach()
        output = linear_clf(feature)
        loss = nn.CrossEntropyLoss()(output, label)
        loss.backward()
        opt_clf.step()
        train_loss += loss.item()
        _, predict = output.max(1)
        total += label.size(0)
        correct += predict.eq(label).sum().item()
        if (ix + 1) % 20 == 0:
            print("L-train loss:{} / L-acc:{}".format(train_loss / (ix + 1),
                                                      100 * correct / total))


def Test_CLF(path):
    model.eval()
    linear_clf.eval()
    total = 0
    correct = 0
    test_loss = 0.0
    global best_acc
    print("========== [Supervised Testing] ==========")
    with torch.no_grad():
        for ix, (sample1, sample2) in enumerate(Linear_testloader):
            data, label = sample1["image"], sample1["label"]
            data, label = data.to(device), label.to(device)
            feature = model(data)
            output = linear_clf(feature.detach())
            loss = nn.CrossEntropyLoss()(output, label)
            test_loss += loss.item()
            _, predict = output.max(1)
            total += label.size(0)
            correct += predict.eq(label).sum().item()
        print("L-test loss:{} / L-acc:{}".format(test_loss / (ix + 1),
                                                 100 * correct / total))
        if (100 * correct / total) > best_acc:
            best_acc = 100 * correct / total
        print("best test accuracy now is: {}".format(best_acc))


if __name__ == "__main__":
    # ========== [param] ==========
    for arg in vars(args):
        print(arg, '===>', getattr(args, arg))
    lr = args.lr
    batch_size = args.batch
    epoch = args.epoch
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
    record_cnn = {"train_loss": []}
    record_clf = {"train_loss": [],
                  "train_acc": [],
                  "test_loss": [],
                  "test_acc": []}

# ========== [data] ==========
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
        batch_size=512,
        shuffle=True,
        drop_last=True,
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
        batch_size=512,
        shuffle=False,
        num_workers=num_worker
    )

    # ========== [visualize] ==========
    if batch_size >= 64:
        visualize(Linear_trainloader, dir_log + '/' + 'visual.png')

    # ========== [device] =============
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ========== [cnn model] ==========
    ckpt = torch.load(dir_ckpt + '/' + 'best.pt')
    model = ResNet(pretrain=False)
    model.load_state_dict(ckpt["cnn"])
    model.to(device)
    linear_clf = Linear_Classifier(classNum=10)
    linear_clf.load_state_dict(ckpt["clf"])
    linear_clf.to(device)
    # opt_clf = optim.SGD(linear_clf.parameters(),
    #                      lr=1e-2,
    #                      momentum=0.9,
    #                      weight_decay=5e-4
    #                      )
    opt_clf = optim.Adam(linear_clf.parameters(),
                         lr=1e-2,
                         weight_decay=5e-4
                         )

    best_acc = 0.0
    for i in range(1, epoch + 1):
        Train_CLF(ep=i, fine_tune=True)
        Test_CLF(path=dir_ckpt + '/' + "best.pt")
