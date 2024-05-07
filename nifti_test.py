#%%
import nibabel as nib
import numpy as np

# Specify the path to your NIfTI file
root = '/media/dysk_a/jr_buler/WMH/dataverse_files/training/'
# root = '/media/dysk_a/jr_buler/WMH/dataverse_files/'
# mri_file_path = root + "/home/jr_buler/Downloads/test2/1 (M, 2023-11-22).nii.gz"
# mask_file_path = root + "/home/jr_buler/Downloads/test2/1 (M, 2023-11-22)-labels.nii.gz"

mri_file_path = root + "Amsterdam/GE3T/100/pre/FLAIR.nii.gz"
mask_file_path = root + "Amsterdam/GE3T/100/wmh.nii.gz"
# mri_file_path = root + "Singapore/55/pre/FLAIR.nii.gz"
# mask_file_path = root + "Singapore/55/wmh.nii.gz"
# mri_file_path = root + "Utrecht/2/pre/FLAIR.nii.gz"
# mask_file_path = root + "Utrecht/2/wmh.nii.gz"


# Load the NIfTI image
brain = {"image": np.array(nib.load(mri_file_path).get_fdata()),
        "mask" : np.array(nib.load(mask_file_path).get_fdata())}

# percentile = 70
# image = brain['image'][:,:,25].copy()
# image = image/image.max()
# image = scipy.ndimage.median_filter(image, size=30)

# threshold = np.percentile(image, percentile)
# image[image >= threshold] = 1
# image[image < threshold] = 0
# plt.imshow(image)
# plt.imshow(brain['image'][:,:,25])

# %%
# print(nifti_image)
# print('HEADER', nifti_image.header)
# print(nifti_image.header.get_data_dtype())
# img = nifti_image.get_fdata()
# print(img.shape)
# %%

import matplotlib.pyplot as plt
def show_slices(all_slices, selected_slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(2, len(selected_slices))
   for i in range(len(selected_slices)):
       axes[0][i].imshow(all_slices["image"][:, :, selected_slices[i]].T, cmap="gray", origin="lower")
       axes[1][i].imshow(all_slices["mask"][:, :, selected_slices[i]].T, cmap="gray", origin="lower")


img = brain
show_slices(brain, [0, 40, 60])
plt.suptitle("Selected slices with masks")  

# %%
import os

dir = "/media/dysk_a/jr_buler/WMH/dataverse_files/training/"
os.chdir(dir)
print(os.listdir(os.getcwd()))
for folder in os.listdir(os.getcwd()):
    os.chdir(os.path.join(dir, folder))
    print(os.listdir(os.getcwd()))
#         continue
#     for file in os.listdir(os.getcwd()):
#         if file.startswith('.'):
#             continue
#%%
for root, dirs, files in os.walk(dir, topdown=False):
   for name in files:
      f = os.path.join(root, name)
      if f.__contains__('pre/FLAIR.nii') or f.__contains__('wmh.nii'):
        #   print(os.path.join(root, name))
          image = np.array(nib.load(mri_file_path).get_fdata())
          print(image.shape)
#    for name in dirs:
#       print(os.path.join(root, name))# %%

# %%
