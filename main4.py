import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from cam1_pupilCenter import GazeEstimation
from cam2_objCenter import ArucoDetection
from RealSenseConfig import RealSense
from unpack_int_ext import *
import cv2
import cvzone
import numpy as np
import concurrent.futures
import time
import threading
import json
from queue import Queue

MODEL_PATH = 'mediapipe_facemodel/face_landmarker.task'
RESOLUTION = (640, 480)
MARKER_SIZE = 8.1
FPS = 15
###############################
CAM1_ID = '831612073374'
CAM2_ID = '023422071056'
CAM3_ID = '827112071528'
CAM5_ID = '830112071254'
#####################################################################
DATASET_DIR = 'DATASET'
SAMPLES_PER_DIR = 500
CAM1_DIR = os.path.join(DATASET_DIR, 'CAMERA_1')
CAM2_DIR = os.path.join(DATASET_DIR, 'CAMERA_2')
CAM3_DIR = os.path.join(DATASET_DIR, 'CAMERA_3')
#####################################################################
cam1 = GazeEstimation(MODEL_PATH, CAM1_ID, RESOLUTION, FPS)
cam2 = GazeEstimation(MODEL_PATH, CAM2_ID, RESOLUTION, FPS)
cam3 = RealSense(CAM3_ID, RESOLUTION, FPS)
cam5_aruco = ArucoDetection(CAM5_ID, RESOLUTION, FPS)
#####################################################################
# Color set
orange = (14, 94, 255)
white = (255, 255, 255)
green = (0, 255, 0)
black = (0, 0, 0)
pink = (255, 0, 255)
teal = (255, 255, 0)
red = (0, 0, 255)


def new_dir(base_dir):
    capture_id = 1
    while os.path.exists(os.path.join(base_dir, f'capture_{capture_id}')):
        capture_id += 1
    new_dir = os.path.join(base_dir, f'capture_{capture_id}')
    os.makedirs(new_dir, exist_ok=True)
    return new_dir


# Create directories for camera2 and camera3 under dataset directory
COLOR_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'color_CAM1'))
DEPTH_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'depth_CAM1'))
COORDS_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'coords_CAM1'))

COLOR_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'color_CAM2'))
DEPTH_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'depth_CAM2'))
COORDS_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'coords_CAM2'))

COLOR_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'color_CAM3'))
DEPTH_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'depth_CAM3'))
COORDS_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'coords_CAM3'))


def save_image(directory, filename, image):
    cv2.imwrite(os.path.join(directory, filename), image)


def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [obj.tolist()]
    return obj


def save_eye_coords(filename, coords):
    eye_coords_dict = {
        "Left pupil_center 2d": coords[0],
        "Left pupil_center 3d": coords[1],
        "Right pupil_center 2d": coords[2],
        "Right pupil_center 3d": coords[3],
        "Center point 2d": coords[4],
        "Center point 3d": coords[5],
        "Gaze point 2d": [list(coords[6])],
        "Gaze point 3d": coords[7],
        "Gaze vector 2d": coords[8],
        "Gaze vector 3d": coords[9]
    }
    with open(filename, 'w') as file:
        json.dump(eye_coords_dict, file, indent=4, default=convert_to_serializable)


total_sample_count = 0
current_sample_count = 0
frame_count = 0  # Define frame_count
data_queue = Queue()


def save_dataset(color_cam3, depth_cam3, coords_cam3, color_cam2, depth_cam2, coords_cam2, color_cam1, depth_cam1,
                 coords_cam1, frame_count):
    global total_sample_count, current_sample_count
    global COLOR_DIR_CAM1, DEPTH_DIR_CAM1, COORDS_DIR_CAM1
    global COLOR_DIR_CAM2, DEPTH_DIR_CAM2, COORDS_DIR_CAM2
    global COLOR_DIR_CAM3, DEPTH_DIR_CAM3, COORDS_DIR_CAM3

    # Check if we need to switch to a new directory for saving samples
    if current_sample_count >= SAMPLES_PER_DIR:
        current_sample_count = 0
        COLOR_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'color_cam3'))
        DEPTH_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'depth_cam3'))
        COORDS_DIR_CAM3 = new_dir(os.path.join(CAM3_DIR, 'coords_cam3'))

        COLOR_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'color_cam2'))
        DEPTH_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'depth_cam2'))
        COORDS_DIR_CAM2 = new_dir(os.path.join(CAM2_DIR, 'coords_cam2'))

        COLOR_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'color_cam1'))
        DEPTH_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'depth_cam1'))
        COORDS_DIR_CAM1 = new_dir(os.path.join(CAM1_DIR, 'coords_cam1'))

    # Define filenames
    color_file_cam3 = f"{frame_count:05d}.png"
    depth_file_cam3 = f"{frame_count:05d}.png"
    coords_file_cam3 = f"{frame_count:05d}.json"

    color_file_cam2 = f"{frame_count:05d}.png"
    depth_file_cam2 = f"{frame_count:05d}.png"
    coords_file_cam2 = f"{frame_count:05d}.json"

    color_file_cam1 = f"{frame_count:05d}.png"
    depth_file_cam1 = f"{frame_count:05d}.png"
    coords_file_cam1 = f"{frame_count:05d}.json"

    # Save images and eye coordinates for CAM3
    save_image(COLOR_DIR_CAM3, color_file_cam3, color_cam3)
    save_image(DEPTH_DIR_CAM3, depth_file_cam3, depth_cam3)
    save_eye_coords(os.path.join(COORDS_DIR_CAM3, coords_file_cam3), coords_cam3)

    # Save images and eye coordinates for CAM2
    save_image(COLOR_DIR_CAM2, color_file_cam2, color_cam2)
    save_image(DEPTH_DIR_CAM2, depth_file_cam2, depth_cam2)
    save_eye_coords(os.path.join(COORDS_DIR_CAM2, coords_file_cam2), coords_cam2)

    # Save images and eye coordinates for CAM1
    save_image(COLOR_DIR_CAM1, color_file_cam1, color_cam1)
    save_image(DEPTH_DIR_CAM1, depth_file_cam1, depth_cam1)
    save_eye_coords(os.path.join(COORDS_DIR_CAM1, coords_file_cam1), coords_cam1)

    current_sample_count += 1
    total_sample_count += 1


def save_dataset_thread(data_queue):
    while True:
        item = data_queue.get()
        if item is None:  # Stop signal
            break
        camera_id, color_cam3, depth_cam3, coords_cam3, \
            color_cam2, depth_cam2, coords_cam2, \
            color_cam1, depth_cam1, coords_cam1, frame_count = item
        save_dataset(color_cam3, depth_cam3, coords_cam3,
                     color_cam2, depth_cam2, coords_cam2,
                     color_cam1, depth_cam1, coords_cam1, frame_count)
        data_queue.task_done()


def display_fps(image, fps, label):
    cv2.putText(image, f'{label} FPS: {int(fps)}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def display_corners(annot_RGB, corner_2d):
    num_corners = 4
    green = (0, 255, 0)
    red = (0, 0, 255)
    if corner_2d is not None:
        for point in corner_2d:
            # Display the corner points

            cv2.circle(annot_RGB, (int(point[0]), int(point[1])), 5, red, -1)
        for i in range(num_corners):
            start_point = (int(corner_2d[i][0]), int(corner_2d[i][1]))
            end_point = (int(corner_2d[(i + 1) % num_corners][0]), int(corner_2d[(i + 1) % num_corners][1]))
            cv2.line(annot_RGB, start_point, end_point, green, 2)


def display_info(annot_RGB, aruco_RGB, aruco_2d, aruco_2d_trans):
    white = (255, 255, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.circle(annot_mask, aruco_2d, 5, white, -1)
    # text_aruco_2d = f"Aruco2D ({int(aruco_2d[0])}, {int(aruco_2d[1])})"
    # tex = f"Aruco_3D}"
    # cvzone.putTextRect(aruco_mask, text_aruco_2d, (5, 30), 0.6, 1, white, black, font=font, colorB=white)
    # cvzone.putTextRect(aruco_mask, text_aruco_3d, (5, 60), 0.6, 1, white, black, font=font, colorB=white)
    ##############################################################################################################################
    if aruco_2d_trans is not None:
        cv2.circle(annot_RGB, (int(aruco_2d_trans[0]), int(aruco_2d_trans[1])), 5, green, -1)
        # cv2.circle(annot_mask, (int(aruco_2d_trans[0]), int(aruco_2d_trans[1])), 5, green, -1)
        cv2.circle(aruco_RGB, (int(aruco_2d_trans[0]), int(aruco_2d_trans[1])), 5, green, -1)
        # cv2.circle(aruco_mask, (int(aruco_2d_trans[0]), int(aruco_2d_trans[1])), 5, green, -1)
        text_cam1 = f"Transformed ArucoCenter ({int(aruco_2d_trans[0])}, {int(aruco_2d_trans[1])})"
        cvzone.putTextRect(annot_RGB, text_cam1, (5, 470), 0.7, 2, green, black, font=font, colorB=black)

    if aruco_2d is not None:
        text_aruco_2d = f"Aruco2D ({int(aruco_2d[0])}, {int(aruco_2d[1])})"
        cvzone.putTextRect(aruco_RGB, text_aruco_2d, (5, 470), 0.7, 2, white, black, font=font, colorB=black)

def calculate_gaze_vector(start_point_2d, end_point_2d, start_point_3d, end_point_3d):
    if (start_point_2d is not None and end_point_2d is not None and
            start_point_3d is not None and end_point_3d is not None and
            len(start_point_2d) > 0 and len(end_point_2d) > 0 and
            len(start_point_3d) > 0 and len(end_point_3d) > 0):
        try:
            # Calculate the 2D gaze vector
            gaze2d = np.array(end_point_2d) - np.array(start_point_2d)

            # Calculate the 3D gaze vector
            gaze3d = np.array(end_point_3d) - np.array(start_point_3d)

            return gaze2d, gaze3d
        except Exception as e:
            print(f"Error calculating gaze vector: {e}")
            return None, None
    else:
        return None, None


def draw_arrowed_line(image, start_point, end_point, color, thickness=2, arrow_length=5, arrow_angle=np.pi / 6):
    # Calculate arrow endpoints
    vec = np.subtract(end_point, start_point)
    vec_norm = np.linalg.norm(vec)
    if vec_norm == 0:
        return

    vec_unit = vec / vec_norm

    # Arrowhead parameters
    arrow_tip = end_point - vec_unit * arrow_length
    arrow_tip_int = tuple(arrow_tip.astype(int))
    # Draw the main line
    cv2.line(image, start_point, arrow_tip_int, color, thickness)

    # Calculate points for a triangular arrowhead
    angle_rad = np.arctan2(vec_unit[1], vec_unit[0])
    angle_left = angle_rad + np.pi - arrow_angle
    angle_right = angle_rad + np.pi + arrow_angle
    arrow_left = arrow_tip + arrow_length * np.array([np.cos(angle_left), np.sin(angle_left)])
    arrow_right = arrow_tip + arrow_length * np.array([np.cos(angle_right), np.sin(angle_right)])

    # Convert points to tuples of integers
    arrow_tip = tuple(arrow_tip.astype(int))
    arrow_left = tuple(arrow_left.astype(int))
    arrow_right = tuple(arrow_right.astype(int))
    points = np.array([arrow_tip, arrow_left, arrow_right])
    cv2.drawContours(image, [points], 0, color, thickness=cv2.FILLED)


def draw_gaze_info(annotRgb, cen2d_c, aruco2d_Cam, gaze2d_c, gaze3d_c, cam_mtx):
    if cen2d_c is None or aruco2d_Cam is None or gaze2d_c is None or gaze3d_c is None:
        return
    if len(cen2d_c) > 0:
        start_point = (int(cen2d_c[0]), int(cen2d_c[1]))
        if gaze2d_c is not None:
            end_point2d = (
                int(start_point[0] + gaze2d_c[0][0]),
                int(start_point[1] + gaze2d_c[0][1])
            )
            draw_arrowed_line(annotRgb, end_point2d, start_point, color=(0, 0, 255), thickness=2)

        if gaze3d_c is not None:
            green = (0, 255, 0)
            cv2.circle(annotRgb, start_point, 3, green, -1)
            arrow_length = 100  # Adjust this value as needed

            end_point_homogeneous = np.dot(cam_mtx, gaze3d_c.T)
            end_point_pixel = (
                int((end_point_homogeneous[0] / end_point_homogeneous[2]).item()),
                int((end_point_homogeneous[1] / end_point_homogeneous[2]).item())
            )
            # Calculate direction vector
            direction_vector = np.array([end_point_pixel[0] - start_point[0], end_point_pixel[1] - start_point[1]])
            direction_norm = np.linalg.norm(direction_vector)
            if direction_norm > 0:
                normalized_direction = direction_vector / direction_norm
            else:
                normalized_direction = direction_vector

            # Calculate endpoint for the arrow
            arrow_tip = np.array(start_point) + arrow_length * normalized_direction
            draw_arrowed_line(annotRgb, start_point, tuple(arrow_tip.astype(int)), color=green, thickness=2)


def process_and_transform_markers(color_cam5, cam5_aruco, K4, D4, R43, T43, K3, D3,
                                  R32, T32, K2, D2, R21, T21, K1, D1,
                                  marker_size):
    # Define target IDs
    target_ids = [1, 2, 3, 4]

    # Dictionary to store marker data
    markers_data = {}

    # Detect markers and estimate pose
    for target_id in target_ids:
        marker_corners, marker_IDs = cam5_aruco.detect_markers(color_cam5, target_id=target_id)
        if marker_corners and marker_IDs:
            _, aruco2d_Cam5, aruco3d_Cam5, _, _, _ = cam5_aruco.estimate_pose(
                marker_corners, marker_IDs, marker_size, color_cam5, K3, D3
            )
            markers_data[target_id] = aruco3d_Cam5[0][0] if aruco3d_Cam5 else None
        else:
            markers_data[target_id] = None

    # Transform marker positions to other cameras
    def transform_marker(aruco3d, R, T, K, D):
        return cam5_aruco.transform_center_point(aruco3d, R, T, K, D)

    if markers_data[1]:
        aruco2d_Cam3, aruco3d_Cam3 = transform_marker(markers_data[1], R43, T43, K3, D3)
        if markers_data[2]:
            aruco2d_Cam2, aruco3d_Cam2 = transform_marker(aruco3d_Cam3, R32, T32, K2, D2)
            if markers_data[3]:
                aruco2d_Cam1, aruco3d_Cam1 = transform_marker(aruco3d_Cam2, R21, T21, K1, D1)
            else:
                aruco2d_Cam1, aruco3d_Cam1 = None, None
        else:
            aruco2d_Cam2, aruco3d_Cam2 = None, None
    else:
        aruco2d_Cam3, aruco3d_Cam3 = None, None
        aruco2d_Cam2, aruco3d_Cam2 = None, None
        aruco2d_Cam1, aruco3d_Cam1 = None, None

    # Draw lines connecting the markers in order 1 → 2 → 3 → 4
    def draw_lines(image, points, color=(0, 255, 0), thickness=2):
        for i in range(len(points) - 1):
            if points[i] is not None and points[i + 1] is not None:
                cv2.line(image, tuple(points[i]), tuple(points[i + 1]), color, thickness)

    # Assuming you have an image to draw on
    image_with_lines = color_cam5.copy()

    # Prepare points for drawing lines
    points = [markers_data.get(i) for i in target_ids]

    # Draw lines connecting the markers
    draw_lines(image_with_lines, points)

    return markers_data, image_with_lines


def main():
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0
    TARGET_ID = 0
    global frame_count
    threading.Thread(target=save_dataset_thread, args=(data_queue,), daemon=True).start()

    try:
        while True:
            with concurrent.futures.ThreadPoolExecutor() as executor:

                future_cam1 = executor.submit(cam1.getFrames)
                future_cam2 = executor.submit(cam2.getFrames)
                future_cam3 = executor.submit(cam3.get_frames)
                future_aruco = executor.submit(cam5_aruco.get_framesAruco)

                cam1_result = future_cam1.result()
                cam2_result = future_cam2.result()
                cam3_result = future_cam3.result()
                cam5_aruco_result = future_aruco.result()

            if cam5_aruco_result is None or cam2_result is None or cam1_result is None or cam3_result is None:
                continue

            # Unpack gaze estimation results

            depthFrame1, depth_cam1, color_cam1, annotRgb1, coords_cam1 = cam1_result
            lCen2d_c1, lCen3d_c1, rCen2d_c1, rCen3d_c1, cen2d_c1, cen3d_c1 = coords_cam1
            depthFrame2, depth_cam2, color_cam2, annotRgb2, coords_cam2 = cam2_result
            lCen2d_c2, lCen3d_c2, rCen2d_c2, rCen3d_c2, cen2d_c2, cen3d_c2 = coords_cam2
            depth_frame, depth_image, color_cam3 = cam3_result
            coords_cam3 = [lCen2d_c2, lCen3d_c2, rCen2d_c2, rCen3d_c2, cen2d_c2, cen3d_c2]

            _, depth_cam5, color_cam5 = cam5_aruco_result

            ih, iw, ic = color_cam5.shape
            arucoImg = np.zeros((ih, iw, ic), dtype=np.uint8)

            if cen2d_c1 is not None and len(cen2d_c1) > 0:
                cvzone.putTextRect(annotRgb1, f"Centerpoint: {cen2d_c1[0]}", (170, 30), 0.7, 2,
                                   font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=orange, colorR=black)
            if cen2d_c2 is not None and len(cen2d_c2) > 0:
                cvzone.putTextRect(annotRgb2, f"Centerpoint: {cen2d_c2[0]}", (170, 30), 0.7, 2,
                                   font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=green, colorR=black)

            cvzone.putTextRect(annotRgb1, 'CAM1', (560, 27), 0.7, 2,
                               font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=orange, colorR=black)
            cvzone.putTextRect(annotRgb2, 'CAM2', (560, 27), 0.7, 2,
                               font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=green, colorR=black)
            cvzone.putTextRect(color_cam3, 'CAM3', (560, 27), 0.7, 2,
                               font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=green, colorR=black)

            cvzone.putTextRect(color_cam5, 'CAM5', (560, 27), 0.7, 2,
                               font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=teal, colorR=black)
            cvzone.putTextRect(arucoImg, 'CAM5', (560, 27), 0.7, 2,
                               font=cv2.FONT_HERSHEY_SIMPLEX, colorB=black, colorT=teal, colorR=black)

            marker_corners, marker_IDs = cam5_aruco.detect_markers(color_cam5, target_id=TARGET_ID)
            if marker_corners and marker_IDs:
                arucoImg, aruco2d_Cam5, aruco3d_Cam5, objPoints_aruco, rvec, tvec = cam5_aruco.estimate_pose(
                    marker_corners,
                    marker_IDs,
                    MARKER_SIZE,
                    color_cam5, K4, D4)
                #######################################################################################################################
                X = round(aruco3d_Cam5[0][0][0], 1)
                Y = round(aruco3d_Cam5[0][0][1], 1)
                Z = round(aruco3d_Cam5[0][0][2], 1)
                aruco3d_Cam5 = (X, Y, Z)

                ############################  TRANSFORM ARUCO CENTER to CAM4 ,CAM3, CAM2, CAM1  ##############################
                if aruco2d_Cam5 is not None:
                    # aruco2d_Cam1, aruco3d_Cam1 = cam5_aruco.transform_center_point(aruco3d_Cam5, normal_R21, normal_T21, K1,
                    #                                                                D1)
                    print(f"aruco2d_Cam5: {aruco2d_Cam5}")
                    print(f"aruco3d_Cam5: {aruco3d_Cam5}")
                    aruco2d_Cam3, aruco3d_Cam3 = cam5_aruco.transform_center(aruco3d_Cam5, R43, T43, K3, D3)
                    print(f"aruco2d_Cam3: {aruco2d_Cam3}")
                    print(f"aruco3d_Cam3: {aruco3d_Cam3}")

                    aruco2d_Cam2, aruco3d_Cam2 = cam5_aruco.transform_center(aruco3d_Cam3, R32, T32, K2, D2)
                    print(f"aruco2d_Cam2: {aruco2d_Cam2}")
                    print(f"aruco3d_Cam2: {aruco3d_Cam2}")

                    aruco2d_Cam1, aruco3d_Cam1 = cam5_aruco.transform_center(aruco3d_Cam2, R21, T21, K1, D1)


                    if aruco2d_Cam3 is not None:
                        cv2.circle(color_cam3, (int(aruco2d_Cam3[0]), int(aruco2d_Cam3[1])), 5, red, -1)

                    if aruco2d_Cam2 is not None:
                        cv2.circle(annotRgb2, (int(aruco2d_Cam2[0]), int(aruco2d_Cam2[1])), 5, green, -1)

                    if aruco2d_Cam1 is not None:
                        cv2.circle(annotRgb1, (int(aruco2d_Cam1[0]), int(aruco2d_Cam1[1])), 5, orange, -1)

                    corners2d_Cam3 = cam5_aruco.transform_and_project(
                        objPoints_aruco, rvec, tvec, K3, D3,(R43, T43)
                    )
                    display_corners(color_cam3, corners2d_Cam3)

                    corners2d_Cam2 = (cam5_aruco.transform_and_project
                        (objPoints_aruco, rvec, tvec, K2, D2, (R43, T43), (R32, T32)))
                    display_corners(annotRgb2, corners2d_Cam2)

                    corners2d_Cam1 = (cam5_aruco.transform_and_project
                        (objPoints_aruco, rvec, tvec, K1, D1, (R43, T43), (R32, T32), (R21, T21) ))
                    display_corners(annotRgb1, corners2d_Cam1)

                    # Calculate Gaze vector
                    gaze2d_c1, gaze3d_c1 = calculate_gaze_vector(cen2d_c1, aruco2d_Cam1, cen3d_c1, aruco3d_Cam1)
                    gaze2d_c2, gaze3d_c2 = calculate_gaze_vector(cen2d_c2, aruco2d_Cam2, cen3d_c2, aruco3d_Cam2)
                    gaze2d_c3, gaze3d_c3 = calculate_gaze_vector(cen2d_c2, aruco2d_Cam2, cen3d_c2, aruco3d_Cam2)

                    # print('\n')
                    # print('##############')
                    # print(f"Center point 2d cam3: {coords_cam3[4]}")
                    # print(f"aruco2d_Cam3: {aruco2d_Cam3}")
                    # print(f"aruco3d_Cam3: {aruco3d_Cam3}")
                    # print(f"gaze2d_c3: {gaze2d_c3}")
                    # print(f"gaze3d_c3: {gaze3d_c3}")
                    # print('\n')
                    # print('##############')
                    # print(f"Center point 2d cam2: {coords_cam2[4]}")
                    # print(f"aruco2d_Cam2: {aruco2d_Cam2}")
                    # print(f"aruco3d_Cam2: {aruco3d_Cam2}")
                    # print(f"gaze2d_c2: {gaze2d_c2}")
                    # print(f"gaze3d_c2: {gaze3d_c2}")
                    ##################################################################################################
                    # coords_cam3.append(aruco2d_Cam2)
                    # coords_cam3.append(aruco3d_Cam2)
                    # coords_cam3.append(gaze2d_c3)
                    # coords_cam3.append(gaze3d_c3)
                    #
                    #
                    # coords_cam2.append(aruco2d_Cam2)
                    # coords_cam2.append(aruco3d_Cam2)
                    # coords_cam2.append(gaze2d_c2)
                    # coords_cam2.append(gaze3d_c2)
                    #
                    # coords_cam1.append(aruco2d_Cam1)
                    # coords_cam1.append(aruco3d_Cam1)
                    # coords_cam1.append(gaze2d_c1)
                    # coords_cam1.append(gaze3d_c1)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC key
                        data_queue.put(None)
                        break

                    ###############################      SAVE DATASET    #############################################
                    ##################################################################################################
                    if aruco2d_Cam2 is not None and aruco3d_Cam2 is not None and aruco2d_Cam1 is not None :
                        data_queue.put(('CAM3_CAM2_CAM1', color_cam2, depth_cam2, coords_cam2,
                                        color_cam2, depth_cam2, coords_cam2,
                                        color_cam1, depth_cam1, coords_cam1, frame_count))
                    #######################################################################################################################

                    if cen2d_c1 is not None and len(cen2d_c1) > 0:
                        cen2d_c1 = (int(cen2d_c1[0][0]), int(cen2d_c1[0][1]))
                        cv2.circle(annotRgb1, cen2d_c1, 5, (0, 0, 255), -1)
                    aruco2d_Cam1 = (int(aruco2d_Cam1[0]), int(aruco2d_Cam1[1]))

                    if cen2d_c2 is not None and len(cen2d_c2) > 0:
                        cen2d_c2 = (int(cen2d_c2[0][0]), int(cen2d_c2[0][1]))
                        cv2.circle(annotRgb2, cen2d_c2, 5, (0, 0, 255), -1)
                    aruco2d_Cam2 = (int(aruco2d_Cam2[0]), int(aruco2d_Cam2[1]))
                    #######################################################################################################################

                    if gaze2d_c1 is not None and gaze3d_c1 is not None:
                        draw_gaze_info(annotRgb1, cen2d_c1, aruco2d_Cam1, gaze2d_c1, gaze3d_c1, K1)
                        text_cam1 = f"Transformed ArucoCenter ({int(aruco2d_Cam1[0])}, {int(aruco2d_Cam1[1])})"
                        cvzone.putTextRect(annotRgb1, text_cam1, (5, 470), 0.7, 2, orange, black,
                                           font=cv2.FONT_HERSHEY_SIMPLEX,
                                           colorB=black)
                    if gaze2d_c2 is not None and gaze3d_c2 is not None:
                        draw_gaze_info(annotRgb2, cen2d_c2, aruco2d_Cam2, gaze2d_c2, gaze3d_c2, K2)
                        text_cam2 = f"Transformed ArucoCenter ({int(aruco2d_Cam2[0])}, {int(aruco2d_Cam2[1])})"
                        cvzone.putTextRect(annotRgb2, text_cam2, (5, 470), 0.7, 2, green, black,
                                           font=cv2.FONT_HERSHEY_SIMPLEX,
                                           colorB=black)
                    text_cam5 = f"ArucoCenter ({int(aruco2d_Cam5[0])}, {int(aruco2d_Cam5[1])})"
                    cvzone.putTextRect(arucoImg, text_cam5, (5, 470), 0.7, 2, teal, black,
                                       font=cv2.FONT_HERSHEY_SIMPLEX,
                                       colorB=black)

            ####################################################################################################################################
            imgList = [arucoImg, color_cam3, annotRgb2, annotRgb1]
            overview = cvzone.stackImages(imgList, 3, 0.8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cvzone.putTextRect(overview, 'FPS: {}'.format(int(fps)), (10, 30), 0.8, 2, (0, 255, 0), (0, 0, 0),
                               font=font, colorB=(0, 0, 0))
            cv2.imshow("overview", overview)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
            fps_frame_count += 1
            frame_count += 1
            fps_end_time = time.time()
            time_diff = fps_end_time - fps_start_time
            if time_diff >= 1:
                fps = fps_frame_count / time_diff
                fps_frame_count = 0
                fps_start_time = fps_end_time

    finally:
        cam5_aruco.stop()
        cam3.stop()
        cam2.stop()
        cam1.stop()
        data_queue.put(None)  # Signal the saving thread to stop


if __name__ == '__main__':
    main()
