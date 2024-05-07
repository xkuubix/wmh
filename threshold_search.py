# %%
import os
import copy
import uuid
import yaml
import argparse
import neptune
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.cuda
from torch import nn
from torchvision import transforms as T

from choose_NCOS import choose_NCOS
from net_utils import train_net, test_net
from BrainWmhDataset import BrainWmhDataset
from torch.utils.data import random_split, DataLoader
from captum.attr import IntegratedGradients, LayerGradCam, LayerAttribution

#%%
def get_args_parser():
    default = '/home/jr_buler/wmh/config.yml'
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=default,
                        help=help)
    return parser


parser = get_args_parser()
args, unknown = parser.parse_known_args()
with open(args.config_path) as file:
    config = yaml.load(file, Loader=yaml.FullLoader)
#%% CONFIG load

# dataloaders args
train_val_frac = config['data_sets']['split_fraction_train_val']
batch_size = config['training_plan']['parameters']['batch_size']
num_workers = config['data_sets']['num_workers']
class_names = config['data_sets']['class_names']

# image parameters
patch_size = config['data_sets']['patch_size']
overlap_train_val = config['data_sets']['overlap_train_val']
overlap_test = config['data_sets']['overlap_test']
mil_params_train_val = {'patch_size': patch_size,
                        'overlap': overlap_train_val}
mil_params_test = {'patch_size': patch_size,
                   'overlap': overlap_test}

# training parameters
net_ar = config['training_plan']['architectures']['names']
net_ar_dropout = config['training_plan']['architectures']['dropout']
criterion_type = config['training_plan']['criterion']
optimizer_type = config['training_plan']['optim_name']
scheduler_type = config['training_plan']['scheduler']

#paths
train_dir = config['dir']['train']
test_dir = config['dir']['test']

# set seed and device
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

#%%
transform = None
brain_test = BrainWmhDataset(root_dir=test_dir,
                             mil_params=mil_params_test,
                             transforms=None,
                             train=False)
print('Test len', len(brain_test))

#%%
#%%
# gen datsets
transform = T.Compose([#T.RandomAffine(degrees=(0), translate=(0, 0.1)),
                       T.RandomHorizontalFlip(),
                       T.RandomVerticalFlip(),
                    #    T.RandomResizedCrop(size=mil_params['patch_size'],
                    #                        scale=(0.9, 1.0),
                    #                        antialias=True)
                    ])

transform = None


brain_train_val = BrainWmhDataset(root_dir=train_dir,
                                  mil_params=mil_params_train_val,
                                  transforms=transform,
                                  train=False) # false = not randomized masks/classes

total_size = len(brain_train_val)
train_size = int(train_val_frac * total_size)  # x data for training
val_size = total_size - train_size
# val_size = int((1 - train_val_frac) * total_size)   # (1-x) data for validation
brain_train, brain_val = random_split(brain_train_val,
                                      [train_size, val_size])

# del zrobic
brain_val = copy.deepcopy(brain_val)
brain_val.dataset.train = False


brain_test = BrainWmhDataset(root_dir=test_dir,
                             mil_params=mil_params_test,
                             transforms=None,
                             train=False)

print('Train len', len(brain_train))
print('Val len', len(brain_val))
print('Test len', len(brain_test))

#%%
# gen data loaders
train_loader = DataLoader(brain_train,
                          batch_size=batch_size,
                        #   shuffle=True,
                          num_workers=num_workers,
                          collate_fn=None,
                          #   sampler=sampler,
                          pin_memory=True
                          )
val_loader = DataLoader(brain_val,
                        batch_size=batch_size,
                        #   shuffle=True,
                        num_workers=num_workers,
                        collate_fn=None,
                        #   sampler=sampler,
                        pin_memory=True
                        )
test_loader = DataLoader(brain_test,
                         batch_size=batch_size,
                         #   shuffle=True,
                         num_workers=num_workers,
                         collate_fn=None,
                         #   sampler=sampler,
                         pin_memory=True
                         )


data_loaders = {"test": train_loader,
                "val": val_loader,
                "train": test_loader}

dl_sizes = [len(item) for item in [train_loader,
                                   val_loader,
                                   test_loader]]

data_loaders_sizes = {"test": dl_sizes[0],
                      "val": dl_sizes[1],
                      "train": dl_sizes[2]}

#%%
# select NCOS
net, criterion, optimizer, scheduler = choose_NCOS(
    net_ar=net_ar,
    dropout=net_ar_dropout,
    device=device,
    pretrained=True,
    criterion_type=criterion_type,
    optimizer_type=optimizer_type,
    lr=0, wd=0,
    scheduler=scheduler_type)

if 1:
    def deactivate_batchnorm(net):
        if isinstance(net, nn.BatchNorm2d):
            net.track_running_stats = False
            net.running_mean = None
            net.running_var = None
            # net.momentum = 0.01
    net.apply(deactivate_batchnorm)
#%%
std = torch.load(config['dir']['root']
                    + 'neptune_saved_models/'
                    + '13c366a6-88dc-44d5-b56a-8097cdd37b48',
                    map_location=device)
net.load_state_dict(std)
with torch.no_grad():
    i = 0
    metrics = {'pixel_acc': list(),
                'iou': list(),
               'dice': list(),
               'precision': list(),
               'recall': list(),
                'f1_score': list(),
               'specificity': list()}
    th = np.arange(1, 99, 1)
    for thv in th:
        for d_s in brain_train:
            # d_s[0].shape torch.Size([1, slice, patch, channel, h, w)

            # for slice_number in range(d_s[0].shape[0]): # shape 1? dla slice-wise
            for slice_number in range(len(d_s[0])): # shape 1? dla slice-wise

                # slice_number = 10

                # bags = d_s[0].squeeze(dim=0)
                bags = d_s[0]
                mil_slice = bags[slice_number].unsqueeze(0).to(device)  # Prepare the input data
                # mil_slice = bags[slice_number].to(device)  # Prepare the input data

                # bags = bags.permute(1, 0, 2, 3, 4) # slice x patch x channel x h x w
                # masks = masks.permute(1, 0, 2, 3) # slice x patch x channel x h x w

                net.eval()
                output = net(mil_slice)  # (optional) positional embedding: d_s[1]['tile_cords'])
                h, w = d_s[1]['img_size'][0], d_s[1]['img_size'][1]
                mask = d_s[1]['full_mask'].squeeze(dim=0)[slice_number]
                attention_map = torch.zeros(h, w)
                attribution_map = torch.zeros(h, w)
                overlay_patch_count = torch.ones(h, w)
                img_from_patches = torch.ones(3, h, w)

                # # gamil
                if net_ar in ['amil', 'gamil'] :
                    # weights = output[1].reshape(-1)
                    weights = net.A.reshape(-1)
                # # clam
                elif net_ar in ['clam_sb', 'clam_mb']:
                    weights = output[3]
                else:
                    raise(NotImplementedError)


                weights = weights.squeeze()
                tiles = d_s[1]['tiles'].squeeze() # dim 1/0
                h_min, w_min, dh, dw, _, _ = tiles[0]
                score = torch.sigmoid(output).detach().cpu().numpy().squeeze()
                pred = int(score.round())

                # ig = IntegratedGradients(net)  # Initialize IntegratedGradients with your model
                # baseline = torch.zeros_like(mil_slice)  # Define a baseline (usually a tensor of zeros)
                # attributions, _ = ig.attribute(mil_slice, baselines=baseline, return_convergence_delta=True)
                layer_gc = LayerGradCam(net, net.feature_extractor.layer4[-1].conv2)  # Initialize GradCAM with model and target layer 
                attributions = layer_gc.attribute(mil_slice, target=0)  # Compute attributions for (0th) first neuron (and only is present)
                attributions = LayerAttribution.interpolate(attributions, (33, 33))
                attributions = attributions.squeeze().cpu() # 

                for iter, item in enumerate(d_s[1]['patch_id'][slice_number]):
                    h_min, w_min, dh, dw, _, _ = tiles[item] # iter lub item

                    attention_map[h_min:h_min+dh, w_min:w_min+dw] +=\
                        weights[iter].detach().cpu()
                    
                    img_from_patches[:, h_min:h_min+dh, w_min:w_min+dw] +=\
                        bags[slice_number][iter]
                    
                    attributions[iter][attributions[iter]<=0] = 0 # Use only positive attributions

                    attribution_map[h_min:h_min+dh, w_min:w_min+dw] +=\
                        attributions[iter]
                    
                    overlay_patch_count[h_min:h_min+dh, w_min:w_min+dw] += 1

                
                # standardising by overlay_patch_count
                attention_map = torch.div(attention_map, overlay_patch_count)
                img_from_patches = torch.div(img_from_patches, overlay_patch_count)
                attribution_map = torch.div(attribution_map, overlay_patch_count)

                # overlay_patch_count /= torch.max(overlay_patch_count.reshape(-1))
                # attention_map -= torch.mean(attention_map.reshape(-1))
                
                # attention map value min-max scaling
                attention_map /= torch.max(attention_map.reshape(-1))

                # attention_map[attention_map <= 0.8] = 0.
                

                percentile = thv # 95
                image = attribution_map.clone()
                threshold = np.percentile(image, percentile)
                image[image >= threshold] = 1
                image[image < threshold] = 0
                from scipy.ndimage import median_filter
                image = median_filter(image, size=16, mode='constant', cval=0)

                mask = d_s[1]['full_mask'].squeeze(0)[slice_number]
                output = torch.from_numpy(image.reshape(-1))
                target = mask.reshape(-1).float()

                tp = torch.sum(output * target)  # TP
                fp = torch.sum(output * (1 - target))  # FP
                fn = torch.sum((1 - output) * target)  # FN
                tn = torch.sum((1 - output) * (1 - target))  # TN
                eps = 1e-5
                pixel_acc = (tp + tn + eps) / (tp + tn + fp + fn + eps)
                iou = (tp + eps) / (tp + fp + fn + eps)
                dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                precision = (tp + eps) / (tp + fp + eps)
                recall = (tp + eps) / (tp + fn + eps)
                f1_score = (2 * tp + eps) / (2 * tp + fp + fn + eps)
                specificity = (tn + eps) / (tn + fp + eps)

                metrics['pixel_acc'].append(pixel_acc.item())
                metrics['iou'].append(iou.item())
                metrics['dice'].append(dice.item())
                metrics['precision'].append(precision.item())
                metrics['recall'].append(recall.item())
                metrics['f1_score'].append(f1_score.item())
                metrics['specificity'].append(specificity.item())


                if 0:
                    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(40, 10))                
                    # ax1.imshow(d_s[1]['full_image'].squeeze(dim=0)[slice_number], cmap='gray')
                    ax1.imshow(img_from_patches.permute(1,2,0), cmap='gray')
                    ax2.imshow(attribution_map, vmin=0, vmax=attribution_map.max(),cmap='hot')
                    ax3.imshow(d_s[1]['full_mask'].squeeze(dim=0)[slice_number], cmap='gray')
                    ax4.imshow(attention_map, cmap='gray')
                    for ax in [ax1, ax2, ax3, ax4]:
                        ax.axis('off')

                    # # _, pred = torch.max(output[0].reshape(-1, 4), 1)
                    # # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
                    # #     continue

                    s_title = 'Ground truth: ' + str(int(d_s[1]['labels'][slice_number].item()))
        
                    ax1.set_title(s_title, fontsize=60)
                    ax2.set_title("Grad-cam", fontsize=60)
                    ax3.set_title("True mask", fontsize=60)
                    s_title = 'Predicted: ' + str(pred)  # gamil
                    s_title += '\n[Score: {:.2f}]'.format(score)
                    # s_title = 'Predicted: ' + str(int(pred[0][0]))  # clam
                    if int(d_s[1]['labels'][slice_number].item()) == pred:
                        ax4.set_title(s_title, fontsize=60, color='green')
                    else:
                        ax4.set_title(s_title, fontsize=60, color='yellow')
                    # # bce / ce
                    # pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
                    # s_title = "Predicted: " + predictions[int(pred[0])] +\
                    #     '[' + str(score[0][0].__round__(2)) + ']'
                    # # [0][0] clam ds // [0][0][0] ag
                # break
            i += 1
            if i == 1:
                # break
                pass
        print('Threshold:', thv)
        print('Pixel accuracy:', np.mean(metrics['pixel_acc']))
        print('IoU:', np.mean(metrics['iou']))
        print('Dice:', np.mean(metrics['dice']))
        print('Precision:', np.mean(metrics['precision']))
        print('Recall:', np.mean(metrics['recall']))
        print('F1 score:', np.mean(metrics['f1_score']))
        print('Specificity:', np.mean(metrics['specificity']))
        print()

# TO DO: function that takes model, image slice, THRESHOLD, and returns attribution/attention map
# TO DO: optimize THRESHOLD with best matching to ground truth masks from training set
# %%