# %%
import os
import math
import json
import yaml
import argparse
import torch
import numpy as np
import nibabel as nib
import multiprocessing as mp

from torchvision import transforms as T
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from test_data_set_prep import test_non_zero_img_unique

import time
start_time = time.time()


# %%

def get_args_parser():
    yaml_dir = os.getcwd()
    default_config_path = os.path.join(yaml_dir, 'config.yml')
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


#%%
# set seed and device
seed = config['seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")
patch_size = config['data_sets']['patch_size']
# %%

save_path = '/media/dysk_a/jr_buler/WMH/patches'
bags = []
num_patches_percentage = []
num_patches = 200
min_percentage = 10  # (1, 100)
max_percentage = 50 # (1, 100)

json_path = os.path.join(save_path, "bag_info.json")
with open(json_path, "r") as json_file:
    indices = json.load(json_file)

non_zero_indices_train = np.array(indices["non_zero_indices_train"])
zero_indices_train = np.array(indices["zero_indices_train"])

non_zero_indices_val = np.array(indices["non_zero_indices_val"])
zero_indices_val = np.array(indices["zero_indices_val"])

# %%
def generate_percentages(total_sum, num_patches, min_percentage=1, max_percentage=15):
    result = []
    # Calculate the remaining sum needed after zeros
    remaining_sum = num_patches * (total_sum // 100)
    
    # Populate the rest of the list
    while remaining_sum > 0:
        value = np.random.randint(min_percentage, max_percentage)
        if remaining_sum >= value:
            result.append(value)
            remaining_sum -= value
        else:
            remaining_sum = 0

    return result

# Generate the list
result_train = np.array(generate_percentages(total_sum=len(non_zero_indices_train),
                                       num_patches=num_patches,
                                       min_percentage=min_percentage,
                                       max_percentage=max_percentage))
zeros_train = np.zeros((len(result_train)))

result_val = np.array(generate_percentages(total_sum=len(non_zero_indices_val),
                                        num_patches=num_patches,
                                        min_percentage=min_percentage,
                                        max_percentage=max_percentage))
zeros_val = np.zeros((len(result_val)))


print(len(result_train))
print(sum(result_train))
print(len(result_val))
print(sum(result_val))


num_patches_percentage_train = np.append(result_train, zeros_train).astype(int)
np.random.shuffle(num_patches_percentage_train)
num_patches_percentage_train = num_patches_percentage_train.tolist()

num_patches_percentage_val = np.append(result_val, zeros_val).astype(int)
np.random.shuffle(num_patches_percentage_val)
num_patches_percentage_val = num_patches_percentage_val


import pandas as pd
a = pd.DataFrame(num_patches_percentage_train)
print(a.value_counts().sort_index())
a = pd.DataFrame(num_patches_percentage_val)
print(a.value_counts().sort_index())
# %%

def generate_bag(percentage):
    
    global subset, non_zero_indices_train, zero_indices_train, non_zero_indices_val, zero_indices_val
    
    if subset == "train":
        non_zero_indices = non_zero_indices_train
        zero_indices = zero_indices_train
        path = os.path.join(save_path, "train")
    else:
        non_zero_indices = non_zero_indices_val
        zero_indices = zero_indices_val
        path = os.path.join(save_path, "val")

    bag = {"image": [], "mask": [], "label": []}

    num_non_zero_patches = math.ceil((num_patches * percentage) / 100)  # percentage 0-100 
    non_zero_patches = np.random.choice(non_zero_indices, size=num_non_zero_patches, replace=False)
    if percentage != 0:
        non_zero_indices = non_zero_indices[~np.isin(non_zero_indices, non_zero_patches)]
    for i in non_zero_patches:
        img_path = os.path.join(path, "non_zero_masks", f"img_{i}.nii.gz")
        mask_path = os.path.join(path, "non_zero_masks", f"mask_{i}.nii.gz")
        label = 1
        bag["image"].append(img_path)
        bag["mask"].append(mask_path)
        bag["label"].append(label)
    
    num_zero_patches = math.floor((num_patches * (100 - percentage)) / 100)  # percentage 0-100 
    zero_patches = np.random.choice(zero_indices, size=num_zero_patches, replace=True)
    for i in zero_patches:
        img_path = os.path.join(path, "zero_masks", f"img_{i}.nii.gz")
        mask_path = os.path.join(path, "zero_masks", f"mask_{i}.nii.gz")
        label = 0
        bag["image"].append(img_path)
        bag["mask"].append(mask_path)
        bag["label"].append(label)
    
    return bag

# Multiprocessing pool
subset = "train"
with mp.Pool(processes=mp.cpu_count()-2) as pool:
    bags_train = pool.map(generate_bag, num_patches_percentage_train)
test_non_zero_img_unique(bags_train)

subset = "val"
with mp.Pool(processes=mp.cpu_count()-2) as pool:
    bags_val = pool.map(generate_bag, num_patches_percentage_val)
test_non_zero_img_unique(bags_val)



# %% 
# create dataset and dataloader form the bags
# from test_data_set_prep import WMHDataset

class WMHDataset(Dataset):
    def __init__(self, bags, transform=None):
        self.bags = bags
        self.transform = transform

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, idx):
        bag = self.bags[idx]
        img_paths = bag["image"]
        mask_paths = bag["mask"]
        label = int(sum(bag["label"]) > 0)
        images = []
        # masks = []
        for img_path, mask_path in zip(img_paths, mask_paths):
            img = np.asarray(nib.load(img_path).dataobj)
            img = torch.from_numpy(img)
            img = img / img.max()
            # mask = nib.load(mask_path).get_fdata()
            images.append(img)
            # masks.append(mask)
        images = torch.stack(images)
        p, h, w = images.shape
        images = images.unsqueeze(1).expand(p, 3, h, w)  # [patch, c, h, w])

        if self.transform:
            images = self.transform(images)
            # masks = self.transform(masks)
        return images, label

transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(90),
    T.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    T.RandomResizedCrop(patch_size, scale=(0.8, 1.0)),
])
transform_val = None

train_dataset = WMHDataset(bags_train, transform=transform_train)
val_dataset = WMHDataset(bags_val, transform=transform_val)

# %%
import pickle
save_path = '/media/dysk_a/jr_buler/WMH/patches'
dataset_path = os.path.join(save_path, "train_dataset.pickle")
with open(dataset_path, 'wb') as output:
    pickle.dump(train_dataset, output)

dataset_path = os.path.join(save_path, "val_dataset.pickle")
with open(dataset_path, 'wb') as output:
    pickle.dump(val_dataset, output)


# %%
print("--- %.2f seconds ---" % (time.time() - start_time))
# %%
# train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
# for images, labels in train_loader:
#     print(len(images))
    # print(images[0].shape)
    # print(labels)
    # break
# %%
