import os
import numpy as np


# Function to explore the data for a specific case
def explore_data(case):
    """
    Explore the data for a specific case.

    Parameters:
    - case (int): The case number to explore.

    Returns:
    None
    """
    # List of file names for the data files
    file_names = [
        'conversion',
        # 'conversion_corrected'
        'temperature',
        'time',
        'z'
    ]
    cases_dict = {
        1: 'flow_rate_down_interp_state',
        2: 'flow_rate_up_interp_state',
        3: 'temp_cool_down_interp_state',
        4: 'temp_cool_up_interp_state'
    }

    # Construct folder and case name based on the provided case number
    case_folder = 'case_' + str(case)
    case_name = cases_dict[case]

    # Dictionary to store loaded data arrays
    data_dict = {}

    # Get the current working directory
    base_dir = os.getcwd()

    # Print separator for clarity
    print('-'*50)

    # Loop through each file name in the list
    for name in file_names:
        # Construct file path for the current data file
        file_path = os.path.join(base_dir, 'data', 'raw', case_folder,
                                 f'{name}_{case_name}.npy')
        # Load data from the file into a np array and store it in the data dictionary
        data_dict[name] = np.load(file_path)

        # Print the name of the data and its shape
        print(name, '   ', data_dict[name].shape)

        # If it's the time vector, print the last time point
        if name == 'time':
            print('Last time point:', '   ', data_dict[name][-1], 's')

        # If it's the z vector, print the last space point
        if name == 'z':
            print('Last space point:', '   ', data_dict[name][-1], 'm')

        # Print separator for clarity
        print('-'*50)
