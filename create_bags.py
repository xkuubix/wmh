import os
import json
import math
import numpy as np
from tqdm import tqdm
import nibabel as nib
import multiprocessing

from torchvision import transforms as T
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from test_data_set_prep import test_non_zero_img_unique


# %%
save_path = '/media/dysk_a/jr_buler/WMH/patches'
bags = []
num_patches_percentage = []
num_patches = 100
min_percentage = 1  # (1, 100)
max_percentage = 15 # (1, 100)

json_path = os.path.join(save_path, "bag_info.json")
with open(json_path, "r") as json_file:
    indices = json.load(json_file)

non_zero_indices = indices["non_zero_indices"]
zero_indices = indices["zero_indices"]

progress = 0
pbar = tqdm(total=len(non_zero_indices), desc="Progress")
while sum(num_patches_percentage) < len(non_zero_indices):
    if np.random.rand() < 0.5:
        num_patches_percentage.append(math.ceil(np.random.randint(min_percentage, max_percentage) * num_patches / 100))
    else:
        num_patches_percentage.append(0)
    progress = sum(num_patches_percentage) / len(non_zero_indices) * 100
    pbar.update(1)
pbar.close()

if sum(num_patches_percentage) > len(non_zero_indices):
    num_patches_percentage[-1] -= sum(num_patches_percentage) - len(non_zero_indices)

def generate_bag(percentage):
    bag = {"image": [], "mask": [], "label": []}
    global non_zero_indices, zero_indices
    non_zero_patches = np.random.choice(non_zero_indices, size=percentage, replace=False)
    if percentage != 0:
        non_zero_indices = list(set(non_zero_indices) - set(non_zero_patches))
    for i in non_zero_patches:
        img_path = os.path.join(save_path, "non_zero_masks", f"img_{i}.nii.gz")
        mask_path = os.path.join(save_path, "non_zero_masks", f"mask_{i}.nii.gz")
        label = 1
        bag["image"].append(img_path)
        bag["mask"].append(mask_path)
        bag["label"].append(label)
    
    num_zero_patches = math.floor((num_patches * (100 - percentage)) / 100)  # percentage 0-100 
    zero_patches = np.random.choice(zero_indices, size=num_zero_patches, replace=True)
    # zero_indices = list(set(zero_indices) - set(zero_patches))
    for i in zero_patches:
        img_path = os.path.join(save_path, "zero_masks", f"img_{i}.nii.gz")
        mask_path = os.path.join(save_path, "zero_masks", f"mask_{i}.nii.gz")
        label = 0
        bag["image"].append(img_path)
        bag["mask"].append(mask_path)
        bag["label"].append(label)
    
    return bag

pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 2)
bags = pool.map(generate_bag, num_patches_percentage)
pool.close()
pool.join()

test_non_zero_img_unique(bags)

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
            img = nib.load(img_path).get_fdata()
            # mask = nib.load(mask_path).get_fdata()
            images.append(img)
            # masks.append(mask)
        if self.transform:
            images = self.transform(images)
            # masks = self.transform(masks)
        return images, label

transform = T.Compose([
    T.ToTensor()
])

train_dataset = WMHDataset(bags, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)

for images, labels in train_loader:
    print(images.shape)
    print(labels)
    break
# %%