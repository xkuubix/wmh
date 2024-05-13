import os
import random
import torch
import numpy as np
import nibabel as nib
import torch.nn.functional
import torchvision.transforms.functional as TF
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class BrainWmhDataset(torch.utils.data.Dataset):
  
    def __init__(self, root_dir: str, mil_params: dict, transforms=None, train=False):
        self.transforms = transforms
        self.train = train
        self.root = root_dir
        self.brain = {"image": [], "mask": [], "brain_mask": []}
        self.patch_size = mil_params['patch_size']
        self.overlap = mil_params['overlap']
        make_brain_mask = False
        # select image and mask files for WMH 
        for root, dirs, files in os.walk(self.root, topdown=False):
            for name in files:
                f = os.path.join(root, name)
                if f.__contains__('pre/FLAIR.nii'):
                    self.brain['image'].append(f)
                    if make_brain_mask:
                        brain = np.array(nib.load(f).get_fdata())
                        brain = torch.from_numpy(brain)
                        brain = brain.permute(2, 0, 1)

                        kernel_size = 32
                        structuring_element = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.double)
                        # Apply morphological erosion
                        erosion = torch.nn.functional.conv2d(brain.unsqueeze(0).permute(1, 0, 2, 3), structuring_element, padding=kernel_size // 2)

                        # Apply morphological dilation            
                        dilation = torch.nn.functional.conv2d(erosion, structuring_element, padding=kernel_size // 2)
                        dilation=dilation/dilation.max()
                        dilation[dilation<0.2*dilation.max()]=0
                        dilation[dilation>=0.2*dilation.max()]=1
                        brain_mask = dilation.squeeze(1)
                        brain_mask = brain_mask[:, :brain.shape[1], : brain.shape[2]]
                        nifti_file = nib.Nifti1Image(brain_mask.numpy(), np.eye(4))
                        # path_to_save = root + '/brain_mask.nii'
                        pth = os.path.join(root, 'brain_mask.nii')
                        nib.save(nifti_file, pth) 
                        self.brain['brain_mask'].append(pth)
                    else:
                        pth = os.path.join(root, 'brain_mask.nii')
                        self.brain['brain_mask'].append(pth)

                if f.__contains__('wmh.nii'):
                    self.brain['mask'].append(f)

    def __getitem__(self, idx):
        # get single brain slices, convert to torch tensor
        mri_file_path = self.brain['image'][idx]
        mask_file_path = self.brain['mask'][idx]
        brain_mask_file_path = self.brain['brain_mask'][idx]
        brain = {"image": np.asarray(nib.load(mri_file_path).dataobj),
                 "mask" : np.asarray(nib.load(mask_file_path).dataobj),
                 "brain_mask" : np.asarray(nib.load(brain_mask_file_path).dataobj)
                 }
        brain['image'] = torch.from_numpy(brain['image'])
        brain['mask'] = torch.from_numpy(brain['mask'])
        brain['brain_mask'] = torch.from_numpy(brain['brain_mask'])

        # Generate data required to sample patches from the image
        tiles = self.get_tiles(brain['image'].shape[0],
                               brain['image'].shape[1],
                               self.patch_size,
                               self.patch_size,
                               self.overlap)
        brain['image'] = brain['image'].permute(2, 0, 1)    # shape [slice, h, w]
        brain['mask'] = brain['mask'].permute(2, 0, 1)
        # deleting 'other patology' mask labels
        # mask_bag[mask_bag==2.0] = 0.
        brain['mask'][brain['mask']==2.0] = 0.
        # image normalization
        # brain['image'] = brain['image'] / 4095
        brain['image'] = brain['image'] / brain['image'].max()

        # image to bag conversion
        h, w = brain['image'].shape[1], brain['image'].shape[2] 
        mask_bag, mask_coords = self.convert_img_to_bag(image=brain['mask'],
                                                        tiles=tiles)
        # img_bag, img_coords  = self.convert_img_to_bag(image=brain['image'],
        #                                                tiles=tiles)

        # ------------------- slice sampling
        patch_labels = mask_bag.sum(dim=(2, 3))
        patch_labels[mask_bag.sum(dim=(2, 3))>0] = 1 # n_patches x n_slices
        
        logger.info(f'image shape: {brain["image"].shape}')
        # select only positive slices
        if True:
            columns_to_keep = []
            # import matplotlib.pyplot as plt
            # plt.imshow(patch_labels.T)
            # plt.show()

            for item in range(patch_labels.shape[1]):
                if patch_labels[:, item].sum() > 0:
                    columns_to_keep.append(item)
            # img_bag = img_bag[:, columns_to_keep, :, :]
            mask_bag = mask_bag[:, columns_to_keep, :, :]
            brain['image'] = brain['image'][columns_to_keep, :, :]
            brain['mask'] = brain['mask'][columns_to_keep, :, :]
            brain['brain_mask'] = brain['brain_mask'][columns_to_keep, :, :]

            
            ## patch_labels = patch_labels[:, columns_to_keep] #todo
            patch_labels = mask_bag.sum(dim=(2, 3))
            patch_labels[mask_bag.sum(dim=(2, 3))>0] = 1 # [n_patches, n_slices] 
            
            
            # plt.imshow(patch_labels.T)
            # plt.show()
        logger.info(f'image shape (after): {brain["image"].shape}')
        # ------------------- BACKGROUND REMOVAL
        # brain['image'] = torch.mul(brain['image'], brain['brain_mask'])
        # -------------------
        img_bag, img_coords  = self.convert_img_to_bag(image=brain['image'],
                                                       tiles=tiles)
        # img_bag shape: [patch, slice, h, w]

        logger.info(f'patch_labels shape: {patch_labels.shape}')
        # randomize slice label by randomly zeroing image ROIs and masks
        if self.train:
            for item in range(patch_labels.shape[1]): # shape[1] = n_slices            
                if (torch.rand(1).item()  >= 0.5):
                    img_bag[patch_labels[:, item].type(torch.BoolTensor), item, :, :] = 0.
                    # min_id = 0
                    # max_id = len(patch_labels[:,item].type(torch.BoolTensor)) - patch_labels[:,item].type(torch.BoolTensor).sum()
                    # length = patch_labels[:,item].type(torch.BoolTensor).sum()
                    # id_to_change = torch.randint(min_id, max_id, (length, ))
                    # pool_to_change = ~patch_labels[:, item].type(torch.BoolTensor)
                    # img_bag[patch_labels[:, item].type(torch.BoolTensor), item, :, :] = img_bag[pool_to_change, item, :, :][id_to_change]
                    mask_bag[patch_labels[:, item].type(torch.BoolTensor), item, :, :] = 0.
        # ------------------- slice sampling
        
        # patch_labels = mask_bag.sum(dim=(2, 3))
        # patch_labels[mask_bag.sum(dim=(2, 3))>0] = 1 # [patch, slice] 
        # plt.imshow(patch_labels.T)
        # plt.show()

        # augmentations
        if self.train and self.transforms:
             for patch in range(img_bag.shape[0]):
                img_bag[patch] = self.transforms(img_bag[patch])
                for slice in range(img_bag.shape[1]):
                    angle = random.choice([-90., 0., 90., 180.])
                    img_bag[patch, slice, :, :] = TF.rotate(img_bag[patch, slice, :, :].unsqueeze(0), angle)
                #     print(img_bag[patch][slice].shape)
        else:
            pass

        # each slice channel expansion to 3ch (to fit pretrained networks)
        img_bag = img_bag.unsqueeze(2).expand(-1, -1, 3, -1, -1)  # [patch, slice, c, h, w])


        # label generation
        slice_wise = True
        if slice_wise:
            img_bag = img_bag.permute(1, 0, 2, 3, 4)   # [slice, patch, c, h, w])
            non_zero_count = mask_bag.sum(dim=(2, 3)).T
        # Z-axis -wise
        else:
            non_zero_count = mask_bag.sum(dim=(2, 3))

        target = 1.0*(non_zero_count.sum(1) > 0)


        # print(img_bag.shape) ~ torch.Size([15, 210, 3, 32, 32])
        img = []
        patch_idx = []
        for slice in range(img_bag.shape[0]):
            patches_to_keep = []
            for patch in range(img_bag.shape[1]):
                # discard patches under certain threshold of non-zero pixels
                if torch.count_nonzero(img_bag[slice, patch, 0, :, :], dim=(0, 1)) >= (0.9*self.patch_size**2):
                    patches_to_keep.append(patch)
            patch_idx.append(patches_to_keep)
            img.append(img_bag[slice, patches_to_keep, :, :, :])

        logger.info(f'img length: {len(img)}')
        logger.info(f'img shape: {img[0].shape}')
        return img, {'labels': target, 'masks': mask_bag,
                     'img_coords': img_coords, 'mask_coords': mask_coords,
                     'img_size': [h, w], 'tiles': tiles, 'patch_id': patch_idx,
                     'full_image': brain['image'],
                     'full_mask': brain['mask']}

    def __len__(self):
        return len(self.brain['image'])
    

    def convert_img_to_bag(self, image, tiles):
        '''
        Utilizes 'tiles' vectors generated by get_tiles function
        Returns set of patches sampled from the whole image
        '''
        hTile = tiles[0][2]
        wTile = tiles[0][3]
        c = image.shape[0]
        img_shape = (len(tiles), c, hTile, wTile)
        new_img = torch.zeros(img_shape)

        for i, tile in enumerate(tiles):
            for channel in range(c):
                new_img[i][channel] = image[channel][tile[0]:tile[0]+tile[2],
                                                     tile[1]:tile[1]+tile[3]]
            
        instances = new_img
        instances_cords = tiles[:, 4:6]

        return instances, instances_cords
    
    def start_points(self, size, split_size, overlap=0):
        points = [0]
        stride = int(split_size * (1-overlap))
        counter = 1
        while True:
            pt = stride * counter
            if pt + split_size >= size:
                points.append(size - split_size)
                break
            else:
                points.append(pt)
            counter += 1
        return points

    def get_tiles(self,
                  h, w,
                  hTile,
                  wTile,
                  overlap):

        '''
        Returns vectors required to sample patches from whole image
        h, w - image height, width
        hTile, wTile - patch height, width
        overlap - value in range [0-1] which stands for overlaying area of
                  surrounding patches from 0 to 100% of theirs area
        '''
        X_points = self.start_points(w, wTile, overlap)
        Y_points = self.start_points(h, hTile, overlap)

        tiles = np.zeros((len(Y_points)*len(X_points), 6), int)

        k = 0
        for i, y_cord in enumerate(Y_points):
            for j, x_cord in enumerate(X_points):
                # split = image[i:i+hTile, j:j+wTile]
                tiles[k] = (y_cord, x_cord, hTile, wTile, i, j)
                k += 1
        return tiles

#%%
#%%
# import matplotlib.pyplot as plt
# for x in range(20):
#     fig, axes = plt.subplots(1, 2)
#     axes[0].imshow(brain[10]['bags'][6][x][0], cmap="gray", origin="lower")
#     axes[1].imshow(brain[10]['masks'][6][x], cmap="gray", origin="lower")