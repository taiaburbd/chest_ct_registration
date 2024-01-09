
import cv2
import numpy as np
import SimpleITK as sitk
import os
import nibabel as nib
from scipy import ndimage as ndi

def fill_chest_cavity(image):
    # Convert the image to 8-bit unsigned integer format
    image = image.astype(np.uint8)
    # Create a zeroed image with the same shape as the input image
    filled_image = np.zeros_like(image)

    # Iterate over each slice in the image
    for i, slice in enumerate(image):
        # Find contours in the image slice
        all_objects, hierarchy = cv2.findContours(slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Create a mask with the same shape as the slice
        mask = np.zeros(slice.shape, dtype="uint8")
        # Calculate the area of each contour
        area = [cv2.contourArea(object_) for object_ in all_objects]

        # Skip if no contours are found
        if len(area) == 0:
            continue

        # Select the contour with the largest area
        index_contour = area.index(max(area))
        # Draw the selected contour on the mask
        cv2.drawContours(mask, all_objects, index_contour, 255, -1)
        # Define a kernel for morphological operations
        kernel = np.ones((7, 7), np.uint8)
        # Apply morphological opening to the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # Store the processed mask in the filled_image array
        filled_image[i, :, :] = mask

    # Normalize and return the filled image
    return filled_image / 255


def remove_gantry(image, segmented):
    # Create a mask for the gantry by selecting the minimum value in segmented
    gantry_mask = segmented * (segmented == np.amin(segmented))
    # Fill the chest cavity in the gantry mask
    contours = fill_chest_cavity(gantry_mask)
    # Multiply the original image with the contours to remove the gantry
    removed = np.multiply(image, contours)

    return removed, contours


def check_fov(img, threshold=-975):
    # Create a copy of the image slice
    copy_img = img.copy()
    # Select a specific slice of the image
    copy_img = copy_img[25, :, :]
    # Get width and height of the slice
    width, height = copy_img.shape
    # Calculate mean intensity of corners of the slice
    top_left_corner = np.mean(copy_img[0:5, 0:5])
    top_right_corner = np.mean(copy_img[0:5, width - 5:width])
    bottom_left_corner = np.mean(copy_img[height - 5:height, 0:5])
    bottom_right_corner = np.mean(copy_img[height - 5:height, width - 5:width])

    # Check if the field of view (FOV) is present in at least three corners
    return int(top_left_corner < threshold) + int(top_right_corner < threshold) + int(bottom_left_corner < threshold)\
           + int(bottom_right_corner < threshold) > 2

def segment_kmeans(image, K=3, attempts=10):
    # Invert the image colors
    image_inv = 255 - image
    # Set criteria for k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # Flatten and normalize the image for k-means processing
    vectorized = image_inv.flatten()
    vectorized = np.float32(vectorized) / 255

    # Apply k-means clustering to the image
    ret, label, center = cv2.kmeans(vectorized, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    # Convert cluster centers to 8-bit unsigned integer format
    center = np.uint8(center * 255)
    # Map the labels to the cluster center values
    res = center[label.flatten()]
    # Reshape the result to the original image shape
    result_image = res.reshape((image.shape))

    return result_image

def image_segmentation(_img_path, output_path,rg_output_path,rg_mask_output_path):
    # Read the original image from the provided file path
    original_img = sitk.ReadImage(_img_path)

    # Rescale the intensity of the image to 8-bit (0 to 255) and cast it to uint8
    img_255 = sitk.Cast(sitk.RescaleIntensity(original_img), sitk.sitkUInt8)

    # Convert the SimpleITK image to a numpy array for processing
    seg_img = sitk.GetArrayFromImage(img_255)

    # Check if the field of view (FOV) is present in the image
    if check_fov(sitk.GetArrayFromImage(original_img)):
        # If FOV is present, perform k-means segmentation
        segmented = segment_kmeans(seg_img)
        print("\nFov presence: True")
    else:
        # If FOV is not present, perform k-means segmentation with K=2
        segmented = segment_kmeans(seg_img, K=2)
        print("\nFov presence: False")

    # Remove gantry artifacts from the segmented image
    removed, gantry_mask = remove_gantry(seg_img, segmented)

    # head, tail = os.path.split(_img_path)

    # Display the original and processed images for comparison

    # Check if 'removed' is already a SimpleITK Image
    if not isinstance(removed, sitk.SimpleITK.Image):
        # If not, convert it from a numpy array to a SimpleITK Image
        removed_sitk = sitk.GetImageFromArray(removed)
    else:
        # If it is, use it as is
        removed_sitk = removed

    if not isinstance(segmented, sitk.SimpleITK.Image):
        # If not, convert it from a numpy array to a SimpleITK Image
        segmentedsitk = sitk.GetImageFromArray(segmented)
    else:
        # If it is, use it as is
        segmentedsitk = segmented
    
    if not isinstance(gantry_mask, sitk.SimpleITK.Image):
        # If not, convert it from a numpy array to a SimpleITK Image
        gantry_mask_sitk = sitk.GetImageFromArray(gantry_mask)
    else:
        # If it is, use it as is
        gantry_mask_sitk = gantry_mask

    # Save the processed image to the specified output path
    sitk.WriteImage(removed_sitk, str(rg_output_path))
    sitk.WriteImage(segmentedsitk, str(output_path))
    sitk.WriteImage(gantry_mask_sitk, str(rg_mask_output_path))

def generate_mask(nifti_file, outputPath):
    # Load the NIfTI file
    image = nib.load(nifti_file)

    # Get the image data
    image_data = image.get_fdata()

    # Apply threshold to create a mask: values between 150 and 255 are set to 0 (black), others are unchanged
    threshold_image = np.where((image_data > 150) & (image_data <= 255), 0, image_data).astype(np.uint8)

    # Morphological operations for segmentation
    struct = np.ones((3, 3, 3))
    segmented = ndi.binary_closing(threshold_image, structure=struct)
    segmented = ndi.binary_opening(segmented, structure=struct)

    # Now, create a new NIfTI image object using the segmented data and the original image's affine
    new_img = nib.Nifti1Image(segmented.astype(np.float32), image.affine)

    # Save the new image to disk
    nib.save(new_img, outputPath)
    
def multiplied(original_nifti_path,segmented_nifti_path,outputPath):

    # Load the original and segmented images
    original_nifti = nib.load(original_nifti_path)
    segmented_nifti = nib.load(segmented_nifti_path)

    # Extract the data arrays
    original_data = original_nifti.get_fdata()
    segmented_data = segmented_nifti.get_fdata()

    # Multiply the original image data by the segmented image data
    # Assuming the segmented image is a binary mask
    multiplied_data = np.multiply(original_data, segmented_data)

    # Now, create a new NIfTI image object using the segmented data and the original image's affine
    new_img = nib.Nifti1Image(multiplied_data.astype(np.float32), original_nifti.affine)

    # Save the new image to disk
    nib.save(new_img, outputPath)

def apply_clahe(nifti_file_path, save_path):
    # Load the NIfTI file
    nifti_image = nib.load(nifti_file_path)

    # Get the image data
    image_data = nifti_image.get_fdata()

    # Normalize the image data to 0-255 range for 8-bit input required by CLAHE
    image_data_normalized = cv2.normalize(image_data, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Create a CLAHE object with specified clipLimit
    clahe = cv2.createCLAHE(clipLimit=0.09)

    # Apply CLAHE to each slice of the image data
    clahe_data = np.zeros_like(image_data_normalized)
    for i in range(image_data_normalized.shape[2]):  # Assuming the third dimension is the slice dimension
        clahe_data[:, :, i] = clahe.apply(image_data_normalized[:, :, i])

    # Create a new NIfTI image from the processed data
    new_nifti_image = nib.Nifti1Image(clahe_data, nifti_image.affine)

    # Save the new NIfTI image
    nib.save(new_nifti_image, save_path)