import cv2
import numpy as np
import glob
import pyrealsense2 as rs
import os
import cvzone
from prettytable import PrettyTable

#######################
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
#######################
def find_chessboard_corners(folder_path, pattern_size, square_size, visualize=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * square_size
    objpoints = []
    imgpoints = []
    images_with_corners = []
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                   f.endswith((".bmp", ".png", ".jpg"))]

    for image_path in image_files:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image {image_path}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            objpoints.append(objp)
            corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            imgpoints.append(corners)
            if visualize:
                img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
                images_with_corners.append(img)
        else:
            print(f"Chessboard not found in image {image_path}")

    return objpoints, imgpoints, images_with_corners


def stack_and_display_images(images1, images2, stack_horizontal=True, display_time=1500):
    for img_x, img_y in zip(images1, images2):
        if stack_horizontal:
            stacked_img = np.hstack((img_x, img_y))
        else:
            stacked_img = np.vstack((img_x, img_y))
        cv2.imshow('CAM 4-3', stacked_img)
        cv2.waitKey(display_time)
    cv2.destroyAllWindows()


def create_combined_table(title, intrinsic_info, extrinsic_info):
    table = PrettyTable()
    #################################################################
    extra_bold = "\033[1m"
    color_red = "\033[91m"
    color_green = "\033[92m"
    color_yellow = "\033[93m"
    end_color = "\033[0m"  # reset to default color
    colored_title = f"{extra_bold}{color_yellow}{title}{end_color}"
    table.title = colored_title
    table.field_names = ["Parameter", "Value"]
    #################################################################
    # Add intrinsic parameters
    table.add_row([f"{extra_bold}{color_green}Intrinsic Parameters{end_color}", ""])
    table.add_row(['-' * 80, '-' * 80])
    for camera, info in intrinsic_info.items():
        matrix = "\n".join(["[" + " ".join(f"{num:.8f}" for num in row) + "]" for row in info["camera_matrix"]])
        # Flatten the list of distortion coefficients if its nested
        distortion_coeffs = info["distortion_coefficients"]
        if isinstance(distortion_coeffs[0], list):
            distortion_coeffs = distortion_coeffs[0]
        distortion = "[" + " ".join(f"{num:.8e}" for num in distortion_coeffs) + "]"
        table.add_row([f"{camera} Matrix", matrix])
        table.add_row(['-' * 80, '-' * 80])
        table.add_row([f"{camera} Dist-Coeffs", distortion])
        table.add_row(['-' * 80, '-' * 80])
    #################################################################
    # Add extrinsic parameters
    table.add_row([f"{extra_bold}{color_red}Extrinsic Parameters{end_color}", ""])
    table.add_row(['-' * 80, '-' * 80])

    for param, value in extrinsic_info.items():
        if isinstance(value, list):
            value_str = "\n".join(["[" + " ".join(f"{num:.8f}" for num in row) + "]" for row in value])
        else:
            value_str = value
        table.add_row([param, value_str])
        table.add_row(['-' * 80, '-' * 80])

    print(table)

def get_builtinIntrinsics(serial_number, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    cam_mtx = np.array([[intrinsics.fx, 0, intrinsics.ppx],
                        [0, intrinsics.fy, intrinsics.ppy],
                        [0, 0, 1]])
    dist_mtx = np.array(intrinsics.coeffs)
    pipeline.stop()

    return cam_mtx, dist_mtx


def calibrate_camera(objpoints, imgpoints, image_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
    retval, cam_mtx, dist_mtx, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size,
                                                                  None, None, criteria=criteria)
    reprojection_error = calculate_reprojection_error(imgpoints, objpoints, cam_mtx, dist_mtx, rvecs, tvecs)

    return cam_mtx, dist_mtx, rvecs, tvecs, reprojection_error


def calculate_reprojection_error(imgpoints, objpoints, cam_mtx, dist_mtx, rvecs, tvecs):
    total_error = 0
    total_points = 0

    for i in range(len(objpoints)):
        imgpoints_proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_mtx, dist_mtx)
        error = cv2.norm(imgpoints[i], imgpoints_proj, cv2.NORM_L2) / len(imgpoints_proj)
        total_error += error * len(objpoints[i])
        total_points += len(objpoints[i])
        reprojection_error = total_error / total_points

    return reprojection_error

def visualize_reprojection_error(image_path_x, imgpoints_x, imgpoints_projx,
                                 image_path_y, imgpoints_y, imgpoints_projy, reprojection_error, display_time=2000):

    img_x = cv2.imread(image_path_x)
    img_y = cv2.imread(image_path_y)

    for pt1, pt2 in zip(imgpoints_x, imgpoints_projx):
        cv2.circle(img_x, (int(pt1[0][0]), int(pt1[0][1])), 5, RED, -1)
        cv2.circle(img_x, (int(pt2[0][0]), int(pt2[0][1])), 3, BLUE, -1)
        cv2.line(img_x, (int(pt1[0][0]), int(pt1[0][1])), (int(pt2[0][0]), int(pt2[0][1])), GREEN, 1)

    for pt1, pt2 in zip(imgpoints_y, imgpoints_projy):
        cv2.circle(img_y, (int(pt1[0][0]), int(pt1[0][1])), 5, RED, -1)
        cv2.circle(img_y, (int(pt2[0][0]), int(pt2[0][1])), 3, BLUE, -1)
        cv2.line(img_y, (int(pt1[0][0]), int(pt1[0][1])), (int(pt2[0][0]), int(pt2[0][1])), GREEN, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    text = f'Reprojection Error: {reprojection_error:.3f}'
    image3_number = os.path.basename(image_path_x)
    image4_number = os.path.basename(image_path_y)
    cvzone.putTextRect(img_x, image3_number, (7, 100), font_scale, thickness, WHITE, BLACK, font)
    cvzone.putTextRect(img_y, image4_number, (7, 100), font_scale, thickness, WHITE, BLACK, font)
    # cvzone.putTextRect(img_x, text, (7, 60), font_scale, thickness, (255,255,255), (0,0,0), font)
    cvzone.putTextRect(img_y, text, (7, 60), font_scale, thickness, WHITE, BLACK, font)
    cvzone.putTextRect(img_x, 'Cam4', (7, 20), font_scale, thickness, RED, BLACK, font)
    cvzone.putTextRect(img_y, 'Cam3', (7, 20), font_scale, thickness, RED, BLACK, font)
    combined_img = np.hstack((img_x, img_y))
    cv2.imshow('Reprojection Error', combined_img)
    cv2.waitKey(display_time)


def calculate_reprojection_error_stereo(objpoints, imgpoints_x, imgpoints_y, cam_mtx_x, dist_mtx_x,
                                        cam_mtx_y, dist_mtx_y, R, T):
    tot_error = 0
    total_points = 0
    camImages_x = "StereoImages/Camera 4"
    camImages_y = "StereoImages/Camera 3"
    global mean_error

    cam_images_x_files = glob.glob(os.path.join(camImages_x, '*.*'))
    cam_images_y_files = glob.glob(os.path.join(camImages_y, '*.*'))

    for i, objpoints in enumerate(objpoints):
        _, rvec_camx, tvec_camx, _ = cv2.solvePnPRansac(objpoints, imgpoints_x[i], cam_mtx_x, dist_mtx_x)
        rp_camx, _ = cv2.projectPoints(objpoints, rvec_camx, tvec_camx, cam_mtx_x, dist_mtx_x)

        if i < len(glob.glob(os.path.join(camImages_x, '*.*'))) and i < len(
                glob.glob(os.path.join(camImages_y, '*.*'))):
            image_path_x = cam_images_x_files[i]
            image_path_y = cam_images_y_files[i]
            tot_error += np.sum(np.square(np.float64(imgpoints_x[i] - rp_camx)))
            total_points += len(objpoints)
            rvec_camy, tvec_camy = cv2.composeRT(rvec_camx, tvec_camx, cv2.Rodrigues(R)[0], T)[:2]
            rp_camy, _ = cv2.projectPoints(objpoints, rvec_camy, tvec_camy, cam_mtx_y, dist_mtx_y)
            tot_error += np.square(np.float64(imgpoints_y[i] - rp_camy)).sum()
            total_points += len(objpoints)
            mean_error = np.sqrt(tot_error / total_points)

            visualize_reprojection_error(image_path_x, imgpoints_x[i], rp_camx, image_path_y, imgpoints_y[i], rp_camy,
                                         mean_error)

    return mean_error


def stereo_calibrate(objpoints, imgpoints_x, imgpoints_y, cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, image_size):

    flags = (cv2.CALIB_FIX_INTRINSIC +
             cv2.CALIB_FIX_PRINCIPAL_POINT +
             cv2.CALIB_ZERO_TANGENT_DIST +
             cv2.CALIB_USE_INTRINSIC_GUESS)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-9)
    rms_error, cam_mtx_xr, dist_mtx_xr, cam_mtx_yr, dist_mtx_yr, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_x, imgpoints_y, cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, image_size,
        criteria=criteria, flags=flags)
    reprojection_error = calculate_reprojection_error_stereo(
        objpoints, imgpoints_x, imgpoints_y, cam_mtx_xr, dist_mtx_xr, cam_mtx_yr, dist_mtx_yr, R, T)

    return cam_mtx_xr, dist_mtx_xr, cam_mtx_yr, dist_mtx_yr, R, T, E, F, reprojection_error


def get_intrinsics_extrinsics_info(cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, R, T, reprojection_error):

    intrinsic_info_chessboard = {
        "CAMERA-4": {
            "camera_matrix": cam_mtx_x.tolist(),
            "distortion_coefficients": dist_mtx_x.tolist()
        },
        "CAMERA-3": {
            "camera_matrix": cam_mtx_y.tolist(),
            "distortion_coefficients": dist_mtx_y.tolist()
        }
    }
    extrinsic_info_chessboard = {
        "Rotation Matrix (R)": R.tolist(),
        "Translation Vector (T)": T.tolist(),
        # "Essential Matrix (E)": E.tolist(),
        # "Fundamental Matrix (F)": F.tolist(),
        "Mean Reprojection Error": reprojection_error
    }
    return intrinsic_info_chessboard, extrinsic_info_chessboard

########################################################################################################################
def main():
    pattern_size = (9, 6)  # (INNER ROWS, INNER COLS)
    image_size = (640, 480)
    square_size = 3.2  # cm
    CAM_4 = '830112071254'
    CAM_3 = '827112071528'
    camImages_4 = 'StereoImages/Camera 4'
    camImages_3 = 'StereoImages/Camera 3'

    objpoints, imgpoints_x, images_with_corners_x = find_chessboard_corners(camImages_4, pattern_size, square_size,
                                                                          visualize=True)
    objpoints, imgpoints_y, images_with_corners_y = find_chessboard_corners(camImages_3, pattern_size, square_size,
                                                                          visualize=True)

    stack_and_display_images(images_with_corners_x, images_with_corners_y, stack_horizontal=True)

    objpoints = np.array(objpoints, dtype=np.float32)
    imgpoints_x = np.array(imgpoints_x, dtype=np.float32)
    imgpoints_y = np.array(imgpoints_y, dtype=np.float32)
    print(f"objpoints: {objpoints.shape}")
    print(f"imgpoints_x: {imgpoints_x.shape}")
    print(f"imgpoints_y: {imgpoints_y.shape}")
    ####################################################################################################################
    # INTRINSIC CALIBRATION USING STEREO IMAGES
    cam_mtx_x, dist_mtx_x, rvecs_x, tvecs_x, reprojection_error_x = calibrate_camera(
                                                                    objpoints, imgpoints_x, image_size)
    cam_mtx_y, dist_mtx_y, rvecs_y, tvecs_y, reprojection_error_y = calibrate_camera(
                                                                    objpoints, imgpoints_y, image_size)
    ####################################################################################################################
    # STEREO CALIBRATION USING STEREO IMAGES (CHESSBOARD)
    cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, R, T, E, F, reprojection_error = stereo_calibrate(
        objpoints, imgpoints_x, imgpoints_y, cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, image_size)
    # Save RRS and TRS to a file
    os.makedirs('stereo_calib_params/stereo_calib', exist_ok=True)

    np.savez('stereo_calib_params/stereo_calib/calib_data43.npz',
             Camera_matrix_4=cam_mtx_x, Distortion_matrix_4=dist_mtx_x,
             Camera_matrix_3=cam_mtx_y, Distortion_matrix_3=dist_mtx_y,
             Rotation=R, Translation=T)

    intrinsic_info_chessboard, extrinsic_info_chessboard = (get_intrinsics_extrinsics_info
                                                            (cam_mtx_x, dist_mtx_x, cam_mtx_y, dist_mtx_y, R, T,
                                                                                          reprojection_error))
    create_combined_table("CHESSBOARD CALIBRATION", intrinsic_info_chessboard, extrinsic_info_chessboard)
    print("Stereo calibration data saved."                                 )

    ####################################################################################################################
    # STEREO CALIBRATION USING REALSENSE INTRINSICS
    cam_mtx_x_RS, dist_mtx_x_RS = get_builtinIntrinsics(CAM_4)
    cam_mtx_y_RS, dist_mtx_y_RS = get_builtinIntrinsics(CAM_3)

    cam_mtx_x_RS, dist_mtx_x_RS, cam_mtx_y_RS, dist_mtx_y_RS, R_RS, T_RS, E_RS, F_RS, reprojection_error_RS \
        = stereo_calibrate(objpoints, imgpoints_x, imgpoints_y, cam_mtx_x_RS, dist_mtx_x_RS,
                                      cam_mtx_y_RS, dist_mtx_y_RS, image_size)
    os.makedirs('stereo_calib_params/stereo_calib_realsense', exist_ok=True)

    np.savez('stereo_calib_params/stereo_calib_realsense/calib_data_RS43.npz',
             Camera_matrix_4=cam_mtx_x_RS, Distortion_matrix_4=dist_mtx_x_RS,
             Camera_matrix_3=cam_mtx_y_RS, Distortion_matrix_3=dist_mtx_y_RS,
             Rotation=R_RS, Translation=T_RS)

    intrinsic_info_RS, extrinsic_info_RS = get_intrinsics_extrinsics_info(cam_mtx_x_RS, dist_mtx_x_RS, cam_mtx_y_RS,
                                                                          dist_mtx_y_RS, R_RS, T_RS,
                                                                          reprojection_error_RS)

    create_combined_table("REALSENSE CALIBRATION", intrinsic_info_RS, extrinsic_info_RS)
    print("Stereo calibration data saved.")
    ####################################################################################################################


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
