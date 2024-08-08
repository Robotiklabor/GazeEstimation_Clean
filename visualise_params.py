import numpy as np
from prettytable import PrettyTable, ALL
from termcolor import colored

def load_calibration_data():
    # Load calibration data for each stereo pair
    calib_54 = np.load('StereoCalib_54/stereo_calib_params/stereo_calib/calib_data54.npz')
    calib_54RS = np.load('StereoCalib_54/stereo_calib_params/stereo_calib_realsense/calib_data_RS54.npz')
    calib_43 = np.load('StereoCalib_43/stereo_calib_params/stereo_calib/calib_data43.npz')
    calib_43RS = np.load('StereoCalib_43/stereo_calib_params/stereo_calib_realsense/calib_data_RS43.npz')
    calib_32 = np.load('StereoCalib_32/stereo_calib_params/stereo_calib/calib_data32.npz')
    calib_32RS = np.load('StereoCalib_32/stereo_calib_params/stereo_calib_realsense/calib_data_RS32.npz')
    calib_21 = np.load('StereoCalib_21/stereo_calib_params/stereo_calib/calib_data21.npz')
    calib_21RS = np.load('StereoCalib_21/stereo_calib_params/stereo_calib_realsense/calib_data_RS21.npz')

    return calib_54, calib_54RS, calib_43, calib_43RS, calib_32, calib_32RS, calib_21, calib_21RS

def create_extrinsics_table():
    # Load calibration data
    calib_54, calib_54RS, calib_43, calib_43RS, calib_32, calib_32RS, calib_21, calib_21RS = load_calibration_data()

    # Create a PrettyTable instance
    table = PrettyTable(hrules=ALL)

    # Define table columns
    table.field_names = ['Stereo Pair', 'R', 'T']

    # Add rows for each stereo pair
    table.add_row(['Cam 5-4 Chess', calib_54['Rotation'], calib_54['Translation']])
    table.add_row(['Cam 5-4 RS', calib_54RS['Rotation'], calib_54RS['Translation']])
    table.add_row(['Cam 4-3 Chess', calib_43['Rotation'], calib_43['Translation']])
    table.add_row(['Cam 4-3 RS', calib_43RS['Rotation'], calib_43RS['Translation']])
    table.add_row(['Cam 3-2 Chess', calib_32['Rotation'], calib_32['Translation']])
    table.add_row(['Cam 3-2 RS', calib_32RS['Rotation'], calib_32RS['Translation']])
    table.add_row(['Cam 2-1 Chess', calib_21['Rotation'], calib_21['Translation']])
    table.add_row(['Cam 2-1 RS', calib_21RS['Rotation'], calib_21RS['Translation']])

    # Set column widths for better visualization
    table.max_width['Stereo Pair'] = 15
    table.max_width['R'] = 60
    table.max_width['T'] = 60

    return table
def create_intrinsics_table():
    # Load calibration data
    calib_54, calib_54RS, calib_43, calib_43RS, calib_32, calib_32RS, calib_21, calib_21RS = load_calibration_data()

    # Create a PrettyTable instance
    table = PrettyTable(hrules=ALL)

    # Define table columns
    table.field_names = ['Stereo Pair', 'K1', 'K2', 'K3', 'K4', 'K5']

    # Add rows for each stereo pair with exactly 6 values
    table.add_row(['Cam 5-4 Chess', '', '', '', calib_54.get('Camera_matrix_4', ''), calib_54.get('Camera_matrix_5', '')])
    table.add_row(['Cam 5-4 RealSense', '', '', '', calib_54RS.get('Camera_matrix_4', ''), calib_54RS.get('Camera_matrix_5', '')])
    table.add_row(['Cam 4-3 Chess', '', '', calib_43.get('Camera_matrix_3', ''), calib_43.get('Camera_matrix_4', ''), ''])
    table.add_row(['Cam 4-3 RealSense', '', '', calib_43RS.get('Camera_matrix_3', ''), calib_43RS.get('Camera_matrix_4', ''), ''])
    table.add_row(['Cam 3-2 Chess', '', calib_32.get('Camera_matrix_2', ''), calib_32.get('Camera_matrix_3', ''), '', ''])
    table.add_row(['Cam 3-2 RealSense', '', calib_32RS.get('Camera_matrix_2', ''), calib_32RS.get('Camera_matrix_3', ''), '', ''])
    table.add_row(['Cam 2-1 Chess', calib_21.get('Camera_matrix_1', ''), calib_21.get('Camera_matrix_2', ''), '', '', ''])
    table.add_row(['Cam 2-1 RealSense', calib_21RS.get('Camera_matrix_1', ''), calib_21RS.get('Camera_matrix_2', ''), '', '', ''])


    # Set column widths for better visualization
    table.max_width['Stereo Pair'] = 18
    table.max_width['K1'] = 60
    table.max_width['K2'] = 60
    table.max_width['K3'] = 60
    table.max_width['K4'] = 60
    table.max_width['K5'] = 60

    return table


# Main execution
if __name__ == "__main__":
    print(colored("''''''''''''''''''''''EXTRINSICS''''''''''''''''''''''", 'red'))
    calibration_table = create_extrinsics_table()
    print(calibration_table)
    print('\n')
    print(colored("''''''''''''''''''''''INTRINSICS''''''''''''''''''''''", 'green'))
    intrinsic_table = create_intrinsics_table()
    print(intrinsic_table)
