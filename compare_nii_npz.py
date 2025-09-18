import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F


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


def convert_nii_to_npz_proper(nii_path, slope=1, intercept=0):
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


def analyze_existing_npz_format(npz_path):
    """Analyze the format of existing NPZ files to understand the conversion"""
    try:
        data = np.load(npz_path)
        print(f"\nAnalyzing existing NPZ: {npz_path}")
        print(f"Available arrays: {list(data.keys())}")

        for key in data.keys():
            array = data[key]
            print(f"{key}: shape={array.shape}, dtype={array.dtype}")
            print(f"  Range: [{array.min():.4f}, {array.max():.4f}]")
            print(f"  Mean: {array.mean():.4f}, Std: {array.std():.4f}")

            # Check if it's normalized
            if array.min() >= -1.1 and array.max() <= 1.1:
                print("  -> Appears to be normalized to [-1, 1] range")

        return data

    except Exception as e:
        print(f"Error analyzing NPZ file: {e}")
        return None


if __name__ == "__main__":
    # File paths
    nii_file = "dataset/dataset/train/train_1/train_1_a/train_1_a_1.nii.gz"
    nii_file = '/home/tp030/CT2Rep/example_download_script/data_volumes/dataset/train_fixed/train_2/train_2_a/train_2_a_1.nii.gz'
    existing_npz = "example_data/CT2Rep/train/1/BGC2074584/_(512, 512).npz"
    output_npz = "/home/tp030/CT2Rep/properly_converted.npz"
    print("=== PROPER NIfTI TO NPZ CONVERSION ===\n")

    # First, analyze the existing NPZ format
    print("1. Analyzing existing NPZ format:")
    analyze_existing_npz_format(existing_npz)

    print("\n" + "=" * 60)

    # Convert with proper formatting
    print("\n2. Converting NIfTI with proper format:")
    img_data = convert_nii_to_npz_proper(nii_file)
    np.savez_compressed(output_npz, arr_0=img_data)

    print("\n" + "=" * 60)
    print("\n3. Verifying converted file:")
    analyze_existing_npz_format(output_npz)
