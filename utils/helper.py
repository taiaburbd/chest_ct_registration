
import nibabel as nib
import numpy as np
import SimpleITK as sitk


def load_image(file_path):
    return sitk.ReadImage(file_path)

def load_landmarks(file_path):
    return np.loadtxt(file_path)

def compute_tre(inhale_image_path, fixed_landmarks, registered_moving_landmarks):
    voxel_spacing = nib.load(inhale_image_path).header.get_zooms()
    fixed_points = voxel_spacing * fixed_landmarks
    moving_points = voxel_spacing * registered_moving_landmarks
    tre = np.linalg.norm(moving_points - fixed_points, axis=1)
    mean_tre = np.mean(tre)
    std_tre = np.std(tre)
    return mean_tre, std_tre

