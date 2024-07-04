# %%
import os
import copy
import uuid
import yaml
import argparse
import neptune
from neptune.utils import stringify_unsupported
import numpy as np
import matplotlib.pyplot as plt
import logging

import torch
import torch.cuda
from torch import nn
from torchvision import transforms as T

from choose_NCOS import choose_NCOS
from net_utils import train_net, test_net
from BrainWmhDataset import BrainWmhDataset, WMHDataset
from torch.utils.data import random_split, DataLoader
# from captum.attr import IntegratedGradients
from captum.attr import LayerGradCam, LayerAttribution

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.ERROR)
#%%
def get_args_parser():
    home_dir = os.path.expanduser("~")
    default_config_path = os.path.join(home_dir, 'wmh', 'config.yml')
    help = '''path to .yml config file
    specyfying datasets/training params'''

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default=default_config_path,
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

#paths
train_dir = config['dir']['train']
test_dir = config['dir']['test']

# training parameters
EPOCHS = config['training_plan']['parameters']['epochs'][0]
PATIENCE = config['training_plan']['parameters']['patience']
lr = config['training_plan']['parameters']['lr'][0]
wd = config['training_plan']['parameters']['wd'][0]
grad_accu = config['training_plan']['parameters']['grad_accu']['is_on']
grad_accu_steps = config['training_plan']['parameters']['grad_accu']['steps']
net_ar = config['training_plan']['architectures']['names']
net_size = config['training_plan']['architectures']['size']
net_ar_dropout = config['training_plan']['architectures']['dropout']
criterion_type = config['training_plan']['criterion']
optimizer_type = config['training_plan']['optim_name']
scheduler_type = config['training_plan']['scheduler']

# set seed and device
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

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
                                  train=True)


# import pickle
# save_path = '/media/dysk_a/jr_buler/WMH/patches'
# dataset_path = os.path.join(save_path, "my_dataset.pickle")
# with open(dataset_path, 'rb') as data:
#     brain_train_val = pickle.load(data)


total_size = len(brain_train_val)
train_size = int(train_val_frac * total_size)  # x data for training
val_size = total_size - train_size
# val_size = int((1 - train_val_frac) * total_size)   # (1-x) data for validation
brain_train, brain_val = random_split(brain_train_val,
                                      [train_size, val_size])

# brain_val = copy.deepcopy(brain_val)
# brain_val.dataset.train = True
# del brain_train_val


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


data_loaders = {"train": train_loader,
                "val": val_loader,
                "test": test_loader,
                }

dl_sizes = [len(item) for item in [train_loader,
                                   val_loader,
                                   test_loader]]

data_loaders_sizes = {"train": dl_sizes[0],
                      "val": dl_sizes[1],
                      "test": dl_sizes[2],
                      }

#%%
# select NCOS
net, criterion, optimizer, scheduler = choose_NCOS(
    net_ar=net_ar,
    net_size=net_size,
    dropout=net_ar_dropout,
    device=device,
    pretrained=True,
    criterion_type=criterion_type,
    optimizer_type=optimizer_type,
    lr=lr, wd=wd,
    scheduler=scheduler_type)


# fine tune
if 0:
    std = torch.load(config['dir']['root']
                     + 'neptune_saved_models/'
                     + '2d111a3b-0b6a-4735-be81-6dae96b39759',
                     map_location=device)
    net.load_state_dict(std)

# in theory batch size is 1 (in practice network sees batch size as concatenated slices)
if 1:
    def deactivate_batchnorm(net):
        if isinstance(net, nn.BatchNorm2d):
            net.track_running_stats = False
            net.running_mean = None
            net.running_var = None
            # net.momentum = 0.01
    net.apply(deactivate_batchnorm)
#%%
# TRAIN NETWORK---------------------------------------------------------------
if config['run_with_neptune']:
    neptune_run = neptune.init_run(project='ProjektMMG/WMH')
    neptune_run['config'] = stringify_unsupported(config)
else:
    neptune_run = None

if 0:
    state_dict_BL, state_dict_BACC, loss_stats, accuracy_stats = train_net(
        net, data_loaders,
        data_loaders_sizes,
        device, criterion,
        optimizer, scheduler,
        num_epochs=EPOCHS,
        patience=PATIENCE,
        neptune_run=neptune_run,
        grad_acc_mode=grad_accu,
        accum_steps=grad_accu_steps
        )

    unique_filename1 = str(uuid.uuid4())
    model_save_path = (config['dir']['root'] + 'neptune_saved_models/'
                        + unique_filename1)
    torch.save(state_dict_BACC, model_save_path)

    unique_filename2 = str(uuid.uuid4())
    model_save_path = (config['dir']['root'] + 'neptune_saved_models/'
                        + unique_filename2)
    torch.save(state_dict_BL, model_save_path)
#%%
# TEST NETWORK----------------------------------------------------------------
if 0:
    # test best val accuracy model
    net_BACC = net
    net_BACC.load_state_dict(state_dict_BACC)
    metrics, figures, best_th, roc_auc = test_net(net_BACC, data_loaders,
                                                  class_names, device)

    if neptune_run:
        neptune_run['test/BACC/metrics'].log(metrics['th_05'])
        neptune_run['test/BACC/conf_mx'].upload(figures['cm_th_05'])
        neptune_run['test/BACC/auc_roc'] = roc_auc
        if criterion_type == 'bce':
            neptune_run['test/BACC/th_best'] = best_th
            neptune_run['test/BACC/metrics_th_best'].log(metrics['th_best'])
            neptune_run['test/BACC/conf_mx_th_best'].upload(figures['cm_th_best'])
            neptune_run['test/BACC/roc'].upload(figures['roc'])
        neptune_run['test/BACC/file_name'] = unique_filename1

    del net_BACC

    # test best val loss model
    net_BL = net
    net_BL.load_state_dict(state_dict_BL)
    metrics, figures, best_th, roc_auc = test_net(net_BL, data_loaders,
                                                  class_names, device)

    if neptune_run:
        neptune_run['test/BL/metrics'].log(metrics['th_05'])
        neptune_run['test/BL/conf_mx'].upload(figures['cm_th_05'])
        neptune_run['test/BL/auc_roc'] = roc_auc
        if criterion_type == 'bce':
            neptune_run['test/BL/th_best'] = best_th
            neptune_run['test/BL/metrics_th_best'].log(metrics['th_best'])
            neptune_run['test/BL/conf_mx_th_best'].upload(figures['cm_th_best'])
            neptune_run['test/BL/roc'].upload(figures['roc'])
        neptune_run['test/BL/file_name'] = unique_filename2

    del net_BL

if neptune_run:
    neptune_run.stop()
# %%
if 1:

    std = torch.load(config['dir']['root']
                     + 'neptune_saved_models/'
                    #  + '2d111a3b-0b6a-4735-be81-6dae96b39759',
                    #  + 'debe6bb9-8f7c-4937-a05c-4c82525b5157',
                    #  + 'ee71998c-e395-4c43-8754-092893ba0f58',
                    #  + '54861e2e-e6a0-4c89-a3b1-1bf293f0bb47',
                     + '8ee1293f-428e-46fa-b3fd-68be52754c0a',
                    #  + '34f54771-8112-4403-8226-0a4c3f5b4b76',
                     map_location=device)
    net.load_state_dict(std)
    with torch.no_grad():
        i = 0
        for d_s in brain_test:
            # d_s[0].shape torch.Size([1, slice, patch, channel, h, w)
            d_s = brain_test[10]
            # for slice_number in range(d_s[0].shape[0]): # shape 1? dla slice-wise
            for slice_number in range(len(d_s[0])): # shape 1? dla slice-wise
                
                # slice_number += 30

                # bags = d_s[0].squeeze(dim=0)
                bags = d_s[0]
                mil_slice = bags[slice_number].unsqueeze(0).to(device)  # Prepare the input data
                # mil_slice = bags[slice_number].to(device)  # Prepare the input data

                # bags = bags.permute(1, 0, 2, 3, 4) # slice x patch x channel x h x w
                # masks = masks.permute(1, 0, 2, 3) # slice x patch x channel x h x w

                net.eval()
                output = net(mil_slice)  # (optional) positional embedding: d_s[1]['tile_cords'])
                h, w = d_s[1]['img_size'][0], d_s[1]['img_size'][1]
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
                # print('weights.shape', weights.shape)
                # print('len(tiles)', len(tiles))
                # attention map values+counts aggregation
                # score = torch.sigmoid(output[0]).detach().cpu().numpy()
                score = torch.sigmoid(output).detach().cpu().numpy().squeeze()
                pred = int(score.round())



                # ig = IntegratedGradients(net)  # Initialize IntegratedGradients with your model
                # baseline = torch.zeros_like(mil_slice)  # Define a baseline (usually a tensor of zeros)
                # attributions, _ = ig.attribute(mil_slice, baselines=baseline, return_convergence_delta=True)
                layer_gc = LayerGradCam(net, net.feature_extractor.layer4[-1].conv2)  # Initialize GradCAM with model and target layer 
                # target = 
                attributions = layer_gc.attribute(mil_slice, target=0)  # Compute attributions for (0th) first neuron (and only is present)
                attributions = LayerAttribution.interpolate(attributions, (33, 33))
                attributions = attributions.squeeze().cpu() # 

                # for iter in range(len(tiles)):
                # najpierw norm wag potem agregować dziel mnoż

                # zero out image regions from which patches were taken
                for iter, item in enumerate(d_s[1]['patch_id'][slice_number]):
                    h_min, w_min, dh, dw, _, _ = tiles[item] # iter lub item
                    img_from_patches[:, h_min:h_min+dh, w_min:w_min+dw] = 0.

                for iter, item in enumerate(d_s[1]['patch_id'][slice_number]):
                    h_min, w_min, dh, dw, _, _ = tiles[item] # iter lub item

                    if len(weights.shape) == 0:
                        attention_map[h_min:h_min+dh, w_min:w_min+dw] +=\
                        weights.detach().cpu().item()
                    else:
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
                # mask image regions from patches were NOT taken
                img_from_patches = np.ma.masked_values(img_from_patches.permute(1,2,0), 1)
                
                attribution_map = torch.div(attribution_map, overlay_patch_count)

                # overlay_patch_count /= torch.max(overlay_patch_count.reshape(-1))
                # attention_map -= torch.mean(attention_map.reshape(-1))
                
                # attention map value min-max scaling
                attention_map /= torch.max(attention_map.reshape(-1))


                percentile = 95 # 95
                image = attribution_map.clone()
                # image = attention_map.clone()
                threshold = np.percentile(image, percentile)
                image[image >= threshold] = 1
                image[image < threshold] = 0
                from scipy.ndimage import median_filter
                image = median_filter(image, size=16, mode='constant', cval=0)
                if score.item() < 0.5:
                    image = torch.zeros(h, w)
                
                mask = d_s[1]['full_mask'].squeeze(0)[slice_number]

                # attention_map[attention_map <= 0.8] = 0.

                if 1:
                    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(40, 10))                
                    # ax1.imshow(d_s[1]['full_image'].squeeze(dim=0)[slice_number], cmap='gray')
                    ax1.imshow(img_from_patches, vmin=0, vmax=attribution_map.max(), cmap='gray')
                    ax2.imshow(attribution_map, vmin=0, vmax=attribution_map.max(),cmap='hot')
                    ax3.imshow(mask, cmap='gray')
                    ax4.imshow(attention_map, cmap='gray')
                    ax5.imshow(image, cmap='gray')
                    for ax in [ax1, ax2, ax3, ax4, ax5]:
                        ax.axis('off')

                    title = 'Case: '
                    title += d_s[1]['mri_file_path'].split('/dataverse_files')[-1].split('/pre')[0]
                    title += '   Slice number: ' + str(d_s[1]['slices_taken'][slice_number])
                    fig.suptitle(title, fontsize=30)

                    # # _, pred = torch.max(output[0].reshape(-1, 4), 1)
                    # # if pred != d_s[1]['labels'] or d_s[1]['labels'] == 0:
                    # #     continue

                    s_title = 'Ground truth: ' + str(int(d_s[1]['labels'][slice_number].item()))
                    s_title += '\nPred: ' + str(pred)  # gamil
                    s_title += ' ({:.2f})'.format(score)
                    if round(score.item()) == d_s[1]['labels'][slice_number]:
                        color = 'black'
                    else:
                        color = 'red'
                    ax1.set_title(s_title, fontsize=60, color=color)
                    ax2.set_title("Grad-cam", fontsize=60)
                    ax3.set_title("True mask", fontsize=60)
                    s_title = 'Attention-map'
                    if int(d_s[1]['labels'][slice_number].item()) == pred:
                        ax4.set_title(s_title, fontsize=60)
                    else:
                        ax4.set_title(s_title, fontsize=60)
                    s_title = 'Attribution-map\nwith th/bin'
                    ax5.set_title(s_title, fontsize=60)
                    # # bce / ce
                    # pred = torch.sigmoid(output[0]).detach().cpu().numpy().round()
                    # s_title = "Predicted: " + predictions[int(pred[0])] +\
                    #     '[' + str(score[0][0].__round__(2)) + ']'
                    # # [0][0] clam ds // [0][0][0] ag
                # break
            i += 1
            if i == 1:
                break

# TO DO: function that takes model, image slice, THRESHOLD, and returns attribution/attention map
# TO DO: optimize THRESHOLD with best matching to ground truth masks from training set

# %%
if 0:
    import matplotlib.pyplot as plt
    def show_slices(brain_ds, slice_num):
        """ Function to display row of image slices """
        fig, axes = plt.subplots(brain_ds[1]['img_coords'][-1][0] + 1,
                            brain_ds[1]['img_coords'][-1][1] + 1,
                            figsize=(5, 5),)
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.1, hspace=0.1)

        for k, (i, j) in enumerate(brain_ds[1]['img_coords']):
            # print(i, j, k)
            #    print(brain_ds[0][0][slice_num].permute(1, 2, 0).shape)
            #    print(brain_ds[1]["masks"][0][slice_num].unsqueeze(0).shape)
            axes[brain_ds[1]['img_coords'][-1][0] - i][j].imshow(brain_ds[0][k][slice_num].permute(1, 2, 0),
                                cmap="gray", origin="lower")
            axes[brain_ds[1]['img_coords'][-1][0] - i][j].axis('off')
        #    axes[i][1].imshow(brain_ds[1]["masks"][i][slice_num].unsqueeze(0).permute(1, 2, 0),
        #                      cmap="gray", origin="lower")

    show_slices(brain_train[0], 10)

# %%
# # wyrysować bilans klas w zależności od overlapa? i patch size'a?
# train_dir = "/media/dysk_a/jr_buler/WMH/dataverse_files/training/"
# import numpy as np
# overlap = np.arange(0., 0.9, 0.05)
# positive_bags_count = []
# negative_bags_count = []
# for ov in overlap:
    
#     mil_params = {'patch_size': 64, 'overlap': ov}
#     brain_train_val = BrainWmhDataset(root_dir=train_dir,
#                                     mil_params=mil_params,
#                                     train=True)
#     positive_bags = 0
#     negative_bags = 0
#     for brain in brain_train_val:
#         positive_bags += brain[1]['labels'].sum().item()
#         negative_bags += len(brain[1]['labels']) - brain[1]['labels'].sum().item()
#     print('overlap', ov, end=' ')
#     print('positive bags', positive_bags, end=' ')
#     print('negative bags', negative_bags, end=' ')
#     print('positive to negative ratio', positive_bags/negative_bags, end='\n')
#     positive_bags_count.append(positive_bags)
#     negative_bags_count.append(negative_bags)


# import matplotlib.pyplot as plt
# plt.plot(overlap, negative_bags_count, color='blue', label='Negative')
# plt.plot(overlap, positive_bags_count, color='red', label='Positive')
# plt.legend()
# plt.xlabel('overlap')
# plt.ylabel('bag count')
# plt.show()

# ratio = [positive_bags_count[i]/negative_bags_count[i] for i in range(len(positive_bags_count))]
# plt.plot(overlap, ratio, color='green', label='ratio')
# plt.legend()
# plt.xlabel('overlap')
# plt.ylabel('positive-to-negative')
# plt.show()
