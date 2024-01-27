import matplotlib.pyplot as plt
import SimpleITK as sitk

def display_images_two_column(image_path1, image1_name, image_path2, image2_name, slice = 60):
    # Convert SimpleITK image to NumPy array
    image_array1 = sitk.GetArrayFromImage(image_path1)
    image_array2 = sitk.GetArrayFromImage(image_path2)
    
    # Display axial, sagittal, and coronal slices using matplotlib
    fig, axes = plt.subplots(1, 2, figsize=(6, 10))

    # Display axial slice at index 60
    axes[0].imshow(image_array1[slice, :, :], cmap='gray')
    axes[0].set_title(f'{image1_name}, slice {slice}')
    axes[0].axis('off')

    # Display axial slice at index 60
    axes[1].imshow(image_array2[60, :, :], cmap='gray')
    axes[1].set_title(f'{image2_name}, slice {slice}')
    axes[1].axis('off')

    plt.show()

def load_and_visualize_landmarks(landmark_path1, landmark_path2, title1, title2):
    landmarks1 = []
    landmarks2 = []

    # Load landmarks from the first file
    with open(landmark_path1, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            landmarks1.append((x, y, z))

    # Load landmarks from the second file
    with open(landmark_path2, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            landmarks2.append((x, y, z))

    # Transpose for plotting
    landmarks1 = list(zip(*landmarks1))
    landmarks2 = list(zip(*landmarks2))

    # Create subplots with two columns
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot landmarks from the first file
    axs[0].scatter(landmarks1[0], landmarks1[1], marker='o', label='Landmarks')
    axs[0].set_title(title1)
    axs[0].set_xlabel('RL (Right/Left)')
    axs[0].set_ylabel('AP (Anterior/Posterior)')
    axs[0].legend()

    # Plot landmarks from the second file
    axs[1].scatter(landmarks2[0], landmarks2[1], marker='o', label='Landmarks')
    axs[1].set_title(title2)
    axs[1].set_xlabel('RL (Right/Left)')
    axs[1].set_ylabel('AP (Anterior/Posterior)')
    axs[1].legend()

    plt.show()

def visualize_registration(fixed_image, moving_image, fixed_landmarks, moving_landmarks):
    # Convert SimpleITK images to NumPy arrays
    fixed_array = sitk.GetArrayFromImage(fixed_image)
    moving_array = sitk.GetArrayFromImage(moving_image)

    # Display the images side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(fixed_array[60, :, :], cmap='gray')
    plt.scatter(fixed_landmarks[:, 0], fixed_landmarks[:, 1], fixed_landmarks[:, 2], 
                c='red', marker='o', label='Fixed Landmarks')
    plt.title(f'Fixed Image with Landmarks')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.imshow(moving_array[60, :, :], cmap='gray')
    plt.scatter(moving_landmarks[:, 0], moving_landmarks[:, 1], moving_landmarks[:, 2],
               c='blue', marker='o', label='Moving Landmarks')
    plt.title('Moving Image with Registration Result Landmarks')
    plt.legend()
    plt.axis('off')

    plt.show()