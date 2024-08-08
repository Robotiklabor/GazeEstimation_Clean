import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import pyrealsense2 as rs
import cvzone
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



# FPS variables
fps_start_time = time.time()
fps_frame_count = 0
fps = 0

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = r"C:\Users\majid\PycharmProjects\Research Project - Eye Gaze Estimation\facegaze\face_landmarker.task"

options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False)


class FaceLandmarkDetector:
    def __init__(self, model_path):
        self.result = None
        self.options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self.print_result)
        self.landmarker = vision.FaceLandmarker.create_from_options(self.options)

    def print_result(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        if result.facial_transformation_matrixes:
            self.result = result
        else:
            print('No face detected.')

    def detect(self, mp_image, timestamp_ms):
        self.landmarker.detect_async(mp_image, timestamp_ms)
def draw_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    for face_landmarks in face_landmarks_list:
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    return annotated_image
def draw_arrowed_line(img, pt1, pt2, color, thickness=1, line_type=cv2.LINE_AA, tip_length=0.1):
    cv2.line(img, pt1, pt2, color, thickness, line_type)
    tip_size = np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) * tip_length
    angle = np.arctan2(pt1[1] - pt2[1], pt1[0] - pt2[0])
    p1 = (int(pt2[0] + tip_size * np.cos(angle + np.pi / 4)),
          int(pt2[1] + tip_size * np.sin(angle + np.pi / 4)))
    cv2.line(img, pt2, p1, color, thickness, line_type)
    p2 = (int(pt2[0] + tip_size * np.cos(angle - np.pi / 4)),
          int(pt2[1] + tip_size * np.sin(angle - np.pi / 4)))
    cv2.line(img, pt2, p2, color, thickness, line_type)
def get_3d_point(depth_frame, depth_intrinsics, x, y):

    # print(depth_frame.width)
    # print(depth_frame.height)
    if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height:
        depth = depth_frame.get_distance(x, y)
        if depth == 0:
            return None
        point_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
        return point_3d
    return None
def process_landmarks(frame, result, depth_frame, depth_intrinsics, ih, iw):
    face2d_points = []
    face3d_points = []
    normalized_face2d_points =[]
    # Translate RealSense depth frame center to top-left corner
    depth_image_width = depth_frame.width
    depth_image_height = depth_frame.height
    depth_center_x = depth_image_width // 2
    depth_center_y = depth_image_height // 2
    for lm in result.face_landmarks:
        for idx, landmark in enumerate(lm): #468
            if idx in [1, 33, 463, 263, 133, 468, 473, 78, 308]:

                x, y = int(landmark.x * iw), int(landmark.y * ih)
                face2d_points.append([x, y])
                # Append normalized coordinates to normalized_face2d_points
                normalized_face2d_points.append([landmark.x, landmark.y])
                cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)



                point_3d = get_3d_point(depth_frame, depth_intrinsics, x, y)
                if point_3d is not None:
                    face3d_points.append(point_3d)

    print(f"face2d_points: {len(face2d_points)} "
          f"face3d_points: {len(face3d_points)}")
    print(f"face2d_points: {normalized_face2d_points} "
          f"face3d_points: {face3d_points}")
    return normalized_face2d_points, face3d_points
def solve_pnp(face2d_points, face3d_points, depth_intrinsics):
    # Prepare 3D landmarks in the camera coordinate system
    face3d_points_camera = np.array(face3d_points, dtype=np.float32)

    # Convert 2D points to numpy array
    image_points = np.array(face2d_points, dtype=np.float32)
    # Get the RealSense camera intrinsics
    width = depth_intrinsics.width
    height = depth_intrinsics.height
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy
    ppx = depth_intrinsics.ppx
    ppy = depth_intrinsics.ppy
    coeffs = depth_intrinsics.coeffs

    # Construct the camera matrix
    camera_matrix = np.array([[fx, 0, ppx],
                               [0, fy, ppy],
                               [0, 0, 1]])



    # Use solvePnP to estimate pose
    success, rotation_vector, translation_vector = cv2.solvePnP(face3d_points_camera, image_points, camera_matrix, distCoeffs=None)
    print(f"face3dpoints: {face3d_points_camera.shape}")
    print(f"image_points: {image_points.shape}")
    print(f"coeffs: {coeffs}")
    print(f"success = {success}")
    print(f"rotation_vector = {rotation_vector}")
    print(f"translation_vector = {translation_vector}")
    return rotation_vector, translation_vector
def solve_pnp_ransac(face2d_points, face3d_points, depth_intrinsics):
    # Prepare 3D landmarks in the camera coordinate system
    face3d_points_camera = np.array(face3d_points, dtype=np.float32)

    # Convert 2D points to numpy array
    image_points = np.array(face2d_points, dtype=np.float32)

    # Get the RealSense camera intrinsics
    fx = depth_intrinsics.fx
    fy = depth_intrinsics.fy
    ppx = depth_intrinsics.ppx
    ppy = depth_intrinsics.ppy
    coeffs = depth_intrinsics.coeffs

    # Construct the camera matrix
    camera_matrix = np.array([[fx, 0, ppx],
                              [0, fy, ppy],
                              [0, 0, 1]])

    # Use solvePnPRansac to estimate pose
    _, rotation_vector, translation_vector, _ = cv2.solvePnPRansac(
        face3d_points_camera, image_points, camera_matrix, distCoeffs=None,
        iterationsCount=200, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

    print(f"face3dpoints: {face3d_points_camera.shape}")
    print(f"image_points: {image_points.shape}")
    print(f"coeffs: {coeffs}")
    print(f"success = {success}")
    print(f"rotation_vector = {rotation_vector}")
    print(f"translation_vector = {translation_vector}")
    print(f"inliers: {len(inliers)}")

    return rotation_vector, translation_vector





def main():
    set_realsense_fps = 30
    global fps_start_time, fps_frame_count, fps
    detector = FaceLandmarkDetector(model_path)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, set_realsense_fps)
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, set_realsense_fps)
    profile = pipeline.start(config)
    sensor = profile.get_device().query_sensors()[1]
    # sensor.set_option(rs.option.exposure, 800)
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    hole_filling = rs.hole_filling_filter()
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    print(depth_intrinsics)
    # Get the depth sensor
    # depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    # # Get the depth scale
    # depth_scale = depth_sensor.get_depth_scale()

    # # Print the depth scale
    # print("Depth Scale:", depth_scale)

    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Apply the hole filling filter
        color_image = np.asanyarray(color_frame.get_data())
        filled_depth = hole_filling.process(depth_frame)
        depth_image = np.asanyarray(filled_depth.get_data())
        depth_image_resized = cv2.resize(depth_image, (color_image.shape[1], color_image.shape[0]))
        depth_image_resized_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_resized, alpha=0.1), cv2.COLORMAP_JET)

        # Overlay depth image on color image
        overlay_image = cv2.addWeighted(color_image, 0.5, depth_image_resized_cm, 0.7, 0)
        frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

        frame_timestamp_ms = int(time.time() * 1000)
        detector.detect(mp_image, frame_timestamp_ms)
        result = detector.result








        if result:
            annotated_image = draw_landmarks_on_image(color_image, result)
            ih, iw, ic = frame.shape
            face2d_points, face3d_points = process_landmarks(annotated_image, result, depth_frame, depth_intrinsics, ih, iw)

            # Convert 2D and 3D points to NumPy arrays
            face2d_points_np = np.array(face2d_points, dtype=np.float32)
            face3d_points_np = np.array(face3d_points, dtype=np.float32) * 0.001
            # print(f"face2d_points: {face2d_points_np}")
            # print(f"face3d_points: {face3d_points_np}")

            #####################
            ####  Solve PnP  ####
            #####################
            # Define camera matrix (based on RealSense camera intrinsics)
            camera_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                                      [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                                      [0, 0, 1]])


            dist_coeffs = np.zeros((5, 1))

            # Run solvePnP
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                face3d_points_np, face2d_points_np, camera_matrix, dist_coeffs,
                iterationsCount=200, reprojectionError=2.0, confidence=0.99, flags=cv2.SOLVEPNP_ITERATIVE)

            if len(face2d_points) == len(face3d_points) and len(face3d_points) > 0:
                if success:

                    # Define the length of the direction vector
                    direction_length = 100

                    # Extract rotation and translation vectors
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    translation_vector = tvec.flatten()
                    # Define the direction vector in the camera coordinate system
                    direction_vector_camera = np.array([0, 0, direction_length], dtype=np.float32)
                    # Transform the direction vector to the world coordinate system
                    direction_vector_world = rotation_matrix.dot(direction_vector_camera)

                    # Nose tip in the world coordinate system (assume it is the translation vector)
                    nose_tip_world = translation_vector

                    # The end point of the direction vector in the world coordinate system
                    end_point_world = nose_tip_world + direction_vector_world

                    # Project the 3D world points to 2D image points
                    nose_tip_2d, _ = cv2.projectPoints(nose_tip_world, rvec, translation_vector,
                                                       camera_matrix,
                                                       distCoeffs=None)
                    end_point_2d, _ = cv2.projectPoints(end_point_world, rvec, translation_vector,
                                                        camera_matrix, distCoeffs=None)

                    nose_tip_2d = tuple(nose_tip_2d.ravel().astype(int))
                    end_point_2d = tuple(end_point_2d.ravel().astype(int))

                    # Draw the direction arrow
                    # cv2.line(annotated_image, nose_tip_2d, end_point_2d, (0, 0, 255), thickness=2)

                    # Optionally, convert rotation vector to Euler angles
                    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rotation_matrix)
                    # Extract roll, pitch, and yaw from angles
                    roll = angles[0] * 180 / np.pi  # Convert radians to degrees
                    pitch = angles[1] * 180 / np.pi
                    yaw = angles[2] * 180 / np.pi

                    # # Print or use the results as needed
                    # print('************************************')
                    # print("Rotation Vector (rvec):", rvec)
                    # print("Translation Vector (tvec):", tvec)
                    # print("Rotation Matrix:", rotation_matrix)
                    # print("Translation Vector:", translation_vector)
                    # print("Euler Angles:", angles)
                    # print('************************************')


            # FPS
            fps_frame_count += 1
            if (time.time() - fps_start_time) > 1.0:
                fps = fps_frame_count / (time.time() - fps_start_time)
                fps_start_time = time.time()
                fps_frame_count = 0
            cvzone.putTextRect(annotated_image, 'FPS: {}'.format(int(fps)), (560, 20), 0.5, cv2.FONT_HERSHEY_PLAIN,
                               (0, 0, 255), 1, cv2.LINE_AA)
            image_list = [annotated_image, depth_image_resized_cm,overlay_image]
            stacked_image = cvzone.stackImages(image_list, 3, 1)
            cv2.imshow('RealsenseView', stacked_image)
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    pipeline.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


 #
 #
 # for idx, (point_2d, point_3d) in enumerate(zip(face2d_points, face3d_points)):
 #                if idx == 1:
 #                    cv2.circle(color_image, (int(point_2d[0]), int(point_2d[1])),
 #                               5, (255, 255, 255), -1)
 #
 #                    nose_world_coords, _ = cv2.projectPoints(face3darr[1], rotation_vector,
 #                                                             translation_vector,cam_matrix,
 #                                                             dist_matrix)
 #
 #                    cv2.circle(annotated_image, (int(nose_world_coords[0][0][0]),
 #                                                 int(nose_world_coords[0][0][1])), 5,(0, 0, 255), -1)
 #
 #                    cvzone.putTextRect(annotated_image, f"Nose 2D: ({int(point_2d[0])}, {int(point_2d[1])})", (20, 50),
 #                                       0.5, cv2.FONT_HERSHEY_PLAIN, (0, 0, 255), 1, cv2.LINE_AA)
 #                    cvzone.putTextRect(annotated_image,
 #                                       f"Nose 3D: ({int(nose_world_coords[0][0][0])}, {int(nose_world_coords[0][0][1])})",
 #                                       (20, 100), 0.5, cv2.FONT_HERSHEY_PLAIN, (0, 0, 255), 1, cv2.LINE_AA)
 #
 #
 #
 #
