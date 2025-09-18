import os
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm


def resize_array(array, current_spacing, target_spacing):
    """
    Resize the array to match the target spacing.

    Args:
    array (torch.Tensor): Input array to be resized.
    current_spacing (tuple): Current voxel spacing (z_spacing, xy_spacing, xy_spacing).
    target_spacing (tuple): Target voxel spacing (target_z_spacing, target_x_spacing, target_y_spacing).

    Returns:
    np.ndarray: Resized array.
    """
    # Calculate new dimensions
    original_shape = array.shape[2:]
    scaling_factors = [
        current_spacing[i] / target_spacing[i] for i in range(len(original_shape))
    ]
    new_shape = [
        int(original_shape[i] * scaling_factors[i]) for i in range(len(original_shape))
    ]
    # Resize the array
    resized_array = (
        F.interpolate(array, size=new_shape, mode="trilinear", align_corners=False)
        .cpu()
        .numpy()
    )
    return resized_array


def convert_nii_to_npz_proper(nii_path):
    """
    Convert NIfTI to NPZ format matching the existing dataset format
    """
    print(f"Converting: {nii_path}")

    nii_img = nib.load(str(nii_path))
    img_data = nii_img.get_fdata()
    header = nii_img.header
    # print(header)
    # Get spacing from NIfTI header
    pixdim = header["pixdim"][1:4]  # pixdim[1:4] contains x, y, z spacing
    xy_spacing = pixdim[0]  # x spacing
    z_spacing = pixdim[2]  # z spacing
    print(
        f"Extracted spacing from NIfTI: xy_spacing={xy_spacing}, z_spacing={z_spacing}"
    )

    # Define the target spacing values
    target_x_spacing = 0.75
    target_y_spacing = 0.75
    target_z_spacing = 1.5

    current = (z_spacing, xy_spacing, xy_spacing)
    target = (target_z_spacing, target_x_spacing, target_y_spacing)
    # this is already scaled
    # img_data = slope * img_data - 8192
    hu_min, hu_max = -1000, 1000
    img_data = np.clip(img_data, hu_min, hu_max)

    img_data = img_data.transpose(2, 0, 1)

    tensor = torch.tensor(img_data)
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    img_data = resize_array(tensor, current, target)
    img_data = img_data[0][0]
    img_data = ((img_data) / 1000).astype(np.float32)

    return img_data


## ========================================
## ============CHANGE HERE============
## ========================================
split = "valid"
## ========================================
## ========================================


def process_volume(name, directory_name, saved_dir):
    """Process a single volume - extract this logic into a separate function"""
    folder1 = name.split("_")[0]
    folder2 = name.split("_")[1]
    folder = folder1 + "_" + folder2
    folder3 = name.split("_")[2]
    subfolder = folder + "_" + folder3
    subfolder = directory_name + folder + "/" + subfolder
    fpath = os.path.join(subfolder, name)

    # Create output directory
    output_subfolder = subfolder.replace("data_volumes", saved_dir)
    os.makedirs(output_subfolder, exist_ok=True)

    output_npz = fpath.replace("nii.gz", "npz").replace("data_volumes", saved_dir)

    if os.path.exists(fpath):
        try:
            img_data = convert_nii_to_npz_proper(fpath)
            np.savez_compressed(output_npz, arr_0=img_data)
            return {"status": "success", "name": name}
        except Exception as e:
            return {"status": "error", "name": name, "error": str(e)}
    else:
        return {"status": "missing", "name": name, "path": fpath}


# Main processing code
directory_name = f"data_volumes/dataset/{split}_fixed/"
saved_dir = "data_converted"
os.makedirs(saved_dir, exist_ok=True)
data = pd.read_csv(f"{split}_labels.csv")

failed_files = []
error_files = []

# Option 1: Using submit with ProcessPoolExecutor
with ProcessPoolExecutor(max_workers=24) as executor:  # Adjust max_workers as needed
    # Submit all tasks
    futures = [
        executor.submit(process_volume, name, directory_name, saved_dir)
        for name in data["VolumeName"]
    ]

    # Process results with tqdm
    for future in tqdm(futures, desc="Converting volumes", total=len(futures)):
        result = future.result()

        if result["status"] == "missing":
            failed_files.append(result["path"])
        elif result["status"] == "error":
            error_files.append(f"{result['name']}: {result['error']}")

# Write failed files to log
if failed_files:
    with open(f"{split}_failed.txt", "w") as f:
        for fpath in failed_files:
            f.write(f"{fpath}\n")

# Write error files to log
if error_files:
    with open(f"{split}_errors.txt", "w") as f:
        for error in error_files:
            f.write(f"{error}\n")

print(f"Processed {len(data)} volumes")
print(f"Failed (missing files): {len(failed_files)}")
print(f"Errors during processing: {len(error_files)}")
