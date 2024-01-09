
from tabulate import tabulate
import SimpleITK as sitk
from pathlib import Path
import numpy as np
import re
import itk
import os
from utils.timer_util import Timer
from utils.helper import compute_tre, load_landmarks


MAIN_PATH           = Path('../FinalProject/').resolve()
TRAIN_DATA_PATH     = MAIN_PATH / 'dataset' / 'train'
OUTPUT_DATA_PATH    = MAIN_PATH / 'dataset' / 'output'

def tre_calculate(landmark_path):
    print("Starting to TRE calculate")
    tre = []
    _mean = 0
    _std = 0
    _total = 0 
    landmark_path = sorted(landmark_path.iterdir(), key=lambda x: x.name)

    for i, dir in enumerate(landmark_path):
        id = dir.stem
        exhale_image_path = TRAIN_DATA_PATH / id / f'{id}_eBHCT.nii.gz'
        original_exhale_landmarks_path = TRAIN_DATA_PATH / id / f'{id}_300_eBH_xyz_r1.txt'
        exhale_landmarks_path = dir / "reg_landmark.txt"
          
        inhale_landmarks = load_landmarks(original_exhale_landmarks_path)
        exhale_landmarks = load_landmarks(exhale_landmarks_path)

        calculate_mean, calculate_std = compute_tre(exhale_image_path, inhale_landmarks, exhale_landmarks)

        tre.append([id, f"{calculate_mean:.4f} ± {calculate_std:.4f}"])
        _mean += calculate_mean
        _std += calculate_std
        _total += 1  # Increment the count of processed files

    # Ensure _total is not zero to avoid division by zero error
    if _total > 0:
        avg_mean = _mean / _total
        avg_std = _std / _total
        tre.append(["Mean", f"{avg_mean:.4f} ± {avg_std:.4f}"])
    else:
        tre.append(["Mean", "N/A"])

    print(tabulate(tre, headers=['COPD_Type','Calculated (TRE) :Mean ± STD (mm)'], tablefmt="grid"))
    print("Done")


# convert to landmark
def extract_landmarks (filepath):
    new_landmarks = np.zeros((300, 3))
    current_landmark_file = open(filepath, "r")
    reg_ex = r'OutputIndexFixed = \[([\d.\s\-]+)\]'

    for i, line in enumerate(current_landmark_file):
        
        match_found = re.search(reg_ex, line, re.M)
        j = match_found.group(1).split()
        j = [round(float(c)) for c in j]
        new_landmarks[i,:] = j
        
    return new_landmarks

def save_landmark(outputpoint_path):
    print("Starting to Saving Landmark")
    for i, dir in enumerate(outputpoint_path.iterdir()):
     
        output_point_path = dir / "outputpoints.txt"
        output_name = dir / "reg_landmark.txt"

        print(output_name)

        #  # Extracting only the landmark points and saving it
        transformed_landmarks = extract_landmarks(output_point_path)
        np.savetxt("{}".format(output_name), transformed_landmarks)
        print()
    print("Successfully completed to Saved Landmark")


def elastix_transformix_registration_method(parameter,param_path, dataset_path, _output_data_path, param_name):
    table = []
     # Timer usage
    timer = Timer()
    print("Starting Registration -> Parameter:", param_name)

    sorted_files = sorted(dataset_path.iterdir(), key=lambda x: x.name)
    for i, dir in enumerate(sorted_files):
            # Check if the item is a .DS_Store file or any other file you want to skip
        if dir.name == '.DS_Store':
            continue
        id = dir.stem
        moving_image_path = dir / f'{id}_eBHCT.nii.gz'
        fixed_image_path  = dir / f'{id}_iBHCT.nii.gz'
        inhale_image_path = dir / f'{id}_300_iBH_xyz_r1.txt'

        print("Fixed Image Path",fixed_image_path)
        print("Moving Image Path",moving_image_path)
        print("Inhale landmark Path",inhale_image_path)


        fixed_image = itk.imread(fixed_image_path, itk.F)
        moving_image = itk.imread(moving_image_path, itk.F)
        
        output_path = _output_data_path / param_name / id
        
        # Check if the directory exists. If not, create it.
        if not output_path.exists():
            os.makedirs(output_path)

        # # Read the parameter file
        parameter_object = itk.ParameterObject.New()
        for _param in parameter:
            _addpath = os.path.join(param_path,_param)
            parameter_object.AddParameterFile(_addpath)
        
        timer.start()
        # # Call registration function
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image, moving_image,
            parameter_object=parameter_object,
            log_to_console=False)

        result_point_set = itk.transformix_pointset(
            moving_image, result_transform_parameters,
            fixed_point_set_file_name=str(inhale_image_path),
            output_directory = str(output_path))
        # Stop the timer
        timer.stop()
        # Show total processing time with task name
        _process_time = timer.get_elapsed_time()
        table.append([
            f"{id}",
            f"{_process_time:.2f}",
        ])
    
    print(tabulate(table, headers=['COPD_Type','times (s)'], tablefmt="grid"))
    print('Registration_done')
    save_landmark(_output_data_path / param_name)
    tre_calculate(_output_data_path / param_name)