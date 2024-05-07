import torch
from torch import nn, optim
from torchvision import models
from mil_net_architectures import AttentionMIL
from mil_net_architectures import DSMIL
from mil_net_architectures import MultiAttentionMIL
from mil_net_architectures import GatedMultiAttentionMIL
# from mil_net_architectures import TransMIL
from mil_net_architectures import APE_SAMIL
from mil_net_architectures import CLAM_SB, CLAM_MB
from torch.optim import lr_scheduler


def choose_NCOS(net_ar: str, dropout: bool,
                device: str,
                pretrained: bool,
                criterion_type: str,
                optimizer_type: str,
                lr: float, wd: float, scheduler: str):

    '''
    net_ar: resnet18, resnet50, alexnet, vgg16
    criterion_type: bce, ce
    optimizer_type: sgd, adam
    scheduler_type: lin, [to be added]
    '''

    # SELECT CRITERION
    if criterion_type == 'bce':
        criterion = nn.BCELoss()
        num_out = 1
    elif criterion_type == 'ce':
        criterion = nn.CrossEntropyLoss()
        num_out = 4

    criterion.to(device)

    # SELECT NET ARCHITECTURE
    if net_ar == 'resnet18':
        net = models.resnet18(pretrained=pretrained)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'resnet50':
        net = models.resnet50(pretrained=pretrained)
        num_features = net.fc.in_features
        net.fc = nn.Linear(num_features, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'alexnet':
        net = models.alexnet(pretrained=pretrained)
        net.classifier[4] = nn.Linear(4096, 1024)
        net.classifier[6] = nn.Linear(1024, num_out)
        if torch.cuda.device_count() == 2:
            net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)
        # to bc

    elif net_ar == 'vgg16':
        net = models.vgg16(pretrained=pretrained)
        # to bc

    elif net_ar == 'amil':
        net = AttentionMIL(num_classes=num_out,
                           pretrained=pretrained,
                           gated=False,
                           dropout=dropout)
        # if torch.cuda.device_count() == 2:
        # net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'gamil':
        net = AttentionMIL(num_classes=num_out,
                           pretrained=pretrained,
                           gated=True,
                           dropout=dropout)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'nl_dsmil':
        net = DSMIL(num_classes=num_out,
                    pretrained=pretrained,
                    nonlinear=True,
                    dropout=dropout)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'l_dsmil':
        net = DSMIL(num_classes=num_out,
                    pretrained=pretrained,
                    nonlinear=False)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'mamil':
        net = MultiAttentionMIL(num_classes=num_out,
                                pretrained=pretrained)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'gmamil':
        net = GatedMultiAttentionMIL(num_classes=num_out,
                                     pretrained=pretrained)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    # elif net_ar == 'trans_mil':
    #     net = TransMIL(num_classes=num_out,
    #                    pretrained=pretrained)
    #     # if torch.cuda.device_count() == 2:
    #     #     net = nn.DataParallel(net, device_ids=[0, 1])
    #     net.to(device)

    elif net_ar == 'test':
        net = APE_SAMIL(num_classes=num_out,
                        pretrained=pretrained)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'clam_sb':
        net = CLAM_SB(num_classes=num_out,
                      pretrained=pretrained,
                      dropout=dropout)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    elif net_ar == 'clam_mb':
        net = CLAM_MB(num_classes=num_out,
                      pretrained=pretrained,
                      dropout=dropout)
        # if torch.cuda.device_count() == 2:
        #     net = nn.DataParallel(net, device_ids=[0, 1])
        net.to(device)

    # SELECT OPTIMIZER
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                              weight_decay=wd)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

    # SELECT SCHEDULER
    if scheduler['name'] == 'lin':
        step_szie = scheduler['step_size']
        gamma = scheduler['gamma']
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=step_szie,
                                        gamma=gamma)
    elif scheduler['name'] == 'one-cycle':
        STEPS_PER_EPOCH = scheduler['steps_per_epoch']
        TOTAL_STEPS = scheduler['epochs'] * STEPS_PER_EPOCH
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
                                            total_steps=TOTAL_STEPS)
    elif scheduler['name'] == 'cos-ann':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-5)
    return net, criterion, optimizer, scheduler
