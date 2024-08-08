import numpy as np




# Load calibration data for each stereo pair
calib_54 = np.load('StereoCalib_54/stereo_calib_params/stereo_calib/calib_data54.npz')
calib_54RS = np.load('StereoCalib_54/stereo_calib_params/stereo_calib_realsense/calib_data_RS54.npz')
calib_43 = np.load('StereoCalib_43/stereo_calib_params/stereo_calib/calib_data43.npz')
calib_43RS = np.load('StereoCalib_43/stereo_calib_params/stereo_calib_realsense/calib_data_RS43.npz')
calib_32 = np.load('StereoCalib_32/stereo_calib_params/stereo_calib/calib_data32.npz')
calib_32RS = np.load('StereoCalib_32/stereo_calib_params/stereo_calib_realsense/calib_data_RS32.npz')
calib_21 = np.load('StereoCalib_21/stereo_calib_params/stereo_calib/calib_data21.npz')
calib_21RS = np.load('StereoCalib_21/stereo_calib_params/stereo_calib_realsense/calib_data_RS21.npz')
####################################################################################################
# REALSENSE DEFAULT PARAMS
####################################################################################################
# '''''''Intrinsic parameters'''''''
K5 = calib_54RS['Camera_matrix_5']
D5 = calib_54RS['Distortion_matrix_5']
K4 = calib_43RS['Camera_matrix_4']
D4 = calib_43RS['Distortion_matrix_4']
K3 = calib_43RS['Camera_matrix_3']
D3 = calib_43RS['Distortion_matrix_3']
K2 = calib_32RS['Camera_matrix_2']
D2 = calib_32RS['Distortion_matrix_2']
K1 = calib_21RS['Camera_matrix_1']
D1 = calib_21RS['Distortion_matrix_1']

# '''''''Extrinsic parameters'''''''

# ''''''''''CAMERA 5-4''''''''''
R54 = calib_54RS['Rotation']
T54 = calib_54RS['Translation']

# ''''''''''CAMERA 4-3''''''''''
R43 = calib_43RS['Rotation']
T43 = calib_43RS['Translation']

# ''''''''''CAMERA 3-2''''''''''
R32 = calib_32RS['Rotation']
T32 = calib_32RS['Translation']

# ''''''''''CAMERA 2-1''''''''''
R21 = calib_21RS['Rotation']
T21 = calib_21RS['Translation']
####################################################################################################
# CHESSBOARD PARAMS
####################################################################################################
# '''''''Intrinsic parameters'''''''
K5_chess = calib_54['Camera_matrix_5']
D5_chess = calib_54['Distortion_matrix_5']
K4_chess = calib_43['Camera_matrix_4']
D4_chess = calib_43['Distortion_matrix_4']
K3_chess = calib_43['Camera_matrix_3']
D3_chess = calib_43['Distortion_matrix_3']
K2_chess = calib_32['Camera_matrix_2']
D2_chess = calib_32['Distortion_matrix_2']
K1_chess = calib_21['Camera_matrix_1']
D1_chess = calib_21['Distortion_matrix_1']


# '''''''Extrinsic parameters'''''''

# ''''''''''CAMERA 5-4''''''''''
R54_chess = calib_54['Rotation']
T54_chess = calib_54['Translation']

# ''''''''''CAMERA 4-3''''''''''
R43_chess = calib_43['Rotation']
T43_chess = calib_43['Translation']

# ''''''''''CAMERA 3-2''''''''''
R32_chess = calib_32['Rotation']
T32_chess = calib_32['Translation']

# ''''''''''CAMERA 2-1''''''''''
R21_chess = calib_21['Rotation']
T21_chess = calib_21['Translation']
####################################################################################################





