'''Train CIFAR with PyTorch.'''
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import argparse
from time import time
import logging.config
import logging

from models_cls import *
# from utils import progress_bar
# progress_bar = print
log_config = "logging.conf"
log_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
# os.chdir("on_cls")
logging.config.fileConfig(log_config)
logger = logging.getLogger('mind')
# os.chdir("..")

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
data_base_dir = "D:/cifar-100/"
ckpt_base_dir = "D:/demo_ml/checkpoint/"

global best_acc_epoch
global learning_rate

# Training
def train(epoch):
    global learning_rate
    epoch_start = time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    logger.info('==>  Epoch: %d, use lr=%f' % (epoch, learning_rate))
    
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    for batch_idx, data in enumerate(trainloader, 0):
        # TODO: add tensorboardX
        start_time = time()
        inputs, targets = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        use_time = time() - start_time
        outinfo = "%d %d %d Loss: %.3f | Acc: %.3f%% (%d/%d) in %fs" % (epoch, batch_idx, 
          len(trainloader),train_loss/(batch_idx+1), 100.*correct/total, correct, total, use_time)
        logger.info(outinfo)
    logger.info("==>  Epoch %d time: %f" % (epoch, time()-epoch_start))

# test
def test(epoch):
    global best_acc
    global best_acc_epoch
    global learning_rate
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            logger.info("%d %d Loss: %.3f | Acc: %.3f%% (%d/%d)" % 
                (batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # TODO: add tensorboardX
    # Save checkpoint.
    print('==> Saving..')
    ckpt_folder = ckpt_base_dir + 'ckpt_cifar100_{}/'.format(model_name)
    if not os.path.isdir(ckpt_folder):
        os.mkdir(ckpt_folder)
    
    acc = 100.*correct/total
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
    torch.save(state, ckpt_folder+'ckpt_epoch.pth')

    print("==> epoch: {}, acc: {}, best_acc: {}, best_acc_epoch: {}".format(
        epoch, acc, best_acc, best_acc_epoch
    ))
    if acc > best_acc:
        torch.save(state, ckpt_folder+'ckpt_epoch{}.pth'.format(epoch))
        best_acc = acc
        best_acc_epoch = epoch
    else:
        print("=-------{}".format(best_acc_epoch))
        if (epoch - best_acc_epoch) >= 10:
            # BUG: lr should be global
            learning_rate = learning_rate * 0.1
            best_acc_epoch = epoch
            print("----- change learning_rate to {} due to acc pause".format(learning_rate))


if __name__=="__main__":
    global best_acc_epoch
    global learning_rate
    if torch.cuda.is_available():
        print("[+] ok")
        print("[+] gpu count: %d"%(torch.cuda.device_count()))
    else:
        print("[-] no cuda found")
        exit()

    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    args = parser.parse_args()
    args.resume = None  # "D:/demo_ml/checkpoint/checkpoint_resnet50/ckpt_epoch113.pth"

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    # TODO: add use open_images_v6
    logger.info('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    base_batch_size = 256  
    batch_size = base_batch_size * torch.cuda.device_count()
    trainset = torchvision.datasets.CIFAR100(root=data_base_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=data_base_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=base_batch_size, shuffle=False, num_workers=2)

    # TODO: try efficient-net
    # Model
    logger.info('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    net = ResNet34(num_classes=100)
    # net = ResNet50()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    model_name = "resnet34"
    net = net.to(device)
    if device == 'cuda:0':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    
    # args.resume = "D:/demo_ml/checkpoint/ckpt_cifar100_resnet34/ckpt_epoch27.pth"
    if args.resume is not None:
        # Load checkpoint.
        ckpt_path = args.resume
        logger.info('==> Resuming from checkpoint..')
        assert os.path.exists(ckpt_path), 'Error: checkpoint not found!'
        checkpoint = torch.load(ckpt_path)
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch'] + 1

    # Hyper-parameters
    criterion = nn.CrossEntropyLoss()
    base_learning_rate = args.lr
    learning_rate = base_learning_rate
    best_acc_epoch = start_epoch
    # if epoch <= 200 * 0.15:
    #     learning_rate = base_learning_rate * 1
    # elif epoch <= 200 * 0.5:
    #     learning_rate = base_learning_rate * 0.1
    # else:
    #     learning_rate = base_learning_rate * 0.1 * 0.1

    logger.info("==> Start training")
    for epoch in range(start_epoch, max(start_epoch+50, 200)):
        train(epoch)
        test(epoch)
