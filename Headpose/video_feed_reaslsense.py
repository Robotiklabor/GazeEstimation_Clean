
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import cvzone
from ultralytics import YOLO
import pyrealsense2 as rs




class KalmanFilter:
    def __init__(self, process_noise_scale=100, measurement_noise_scale=100):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * process_noise_scale
        self.kf.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * measurement_noise_scale
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.kf.statePost = np.zeros(4, dtype=np.float32)

    def predict(self, coordx, coordy):
        # Estimate the position of the object
        measured = np.array([[np.float32(coordx)], [np.float32(coordy)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        # x, y = int(predicted[0]), int(predicted[1])
        x, y = predicted[0], predicted[1]

        return x, y






def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

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

#

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# STEP 2: Create an FaceLandmarker object.
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





fps_start_time = time.time()
fps_frame_count = 0
fps = 0
frame_index = 0
set_realsense_fps = 30
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
detector = vision.FaceLandmarker.create_from_options(options)
kf_list_2d = [KalmanFilter() for _ in range(20)]

while True:

    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    frame = np.asanyarray(color_frame.get_data())
    filled_depth = hole_filling.process(depth_frame)
    depth_image = np.asanyarray(filled_depth.get_data())
    depth_image_resized = cv2.resize(depth_image, (frame.shape[1], frame.shape[0]))
    depth_image_resized_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image_resized, alpha=0.1), cv2.COLORMAP_JET)
    overlay_image = cv2.addWeighted(frame, 0.5, depth_image_resized_cm, 0.7, 0)
    frame = cv2.flip(frame, 1)
    depth_image_resized_cm = cv2.flip(depth_image_resized_cm, 1)
    overlay_image = cv2.flip(overlay_image, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = int(time.time() * 1000)
    ih, iw, ic = frame.shape
    landmark_list = []
    face2d= []
    face3d= []
    face_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)

    if not face_landmarker_result.face_landmarks:
        print("No face landmarks detected")
        continue
    for lm in face_landmarker_result.face_landmarks:
        landmark_list.append(lm)
        for idx, landmark in enumerate(landmark_list[0]):

            x,y = int(round(landmark.x*iw)), int(round(landmark.y*ih))
            z = depth_frame.get_distance(int(x), int(y))
            if (idx == 1 or idx == 33 or idx == 463 or idx == 263 or idx == 133 or idx == 468
                    or idx == 473 or idx == 78 or idx == 308 or idx == 0 or idx == 13 or idx == 14
                    or idx == 17 or idx == 61 or idx == 291 or idx == 199
                    or idx == 55 or idx == 107 or idx == 285 or idx == 336):
                    cv2.circle(frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
                    if idx==1:
                        nose2d = (landmark.x*iw,landmark.y*ih)
                        nose3d = (landmark.x*iw,landmark.y*ih,landmark.z*1000)
                        nose_depth = depth_frame.get_distance(int(nose2d[0]), int(nose2d[1]))
                        cvzone.putTextRect(depth_image_resized_cm, "Nose Depth: " + str(round(nose_depth, 2)), (350, 20), 0.5,
                                           cv2.FONT_HERSHEY_PLAIN,
                                           (0, 0, 255), 1, cv2.LINE_AA)
                        print(nose_depth)
                        cv2.circle(depth_image_resized_cm, (int(nose2d[0]), int(nose2d[1])), 4, (0, 0, 255), -1)
                    # Do the same for eye


                    face2d.append([x, y])
                    face3d.append([x, y, landmark.z])
                    smoothed_face2d = []
                    for i, (x, y) in enumerate(face2d):
                        kf = kf_list_2d[i]
                        smoothed_x, smoothed_y = kf.predict(x, y)
                        cv2.circle(frame, (int(smoothed_x[0]), int(smoothed_y[0])), radius=3, color=(255, 0, 0), thickness=4)
                        smoothed_face2d.append([smoothed_x[0], smoothed_y[0]])
                    # smoothed_face2d = [tuple(coord) for coord in smoothed_face2d]

                    smoothed_face3d = []
                    for i, (x, y, z) in enumerate(face3d):
                        kf = kf_list_2d[i]
                        smoothed_x, smoothed_y = kf.predict(x, y)
                        cv2.circle(frame, (int(smoothed_x[0]), int(smoothed_y[0])), radius=3, color=(255, 0, 0),
                                   thickness=4)  # Hollow circle
                        smoothed_face3d.append([float(smoothed_x[0]), float(smoothed_y[0]), float(z)])
            # print(face2d)
            # print('******')
            # print(face3d)
        face2darr = np.array(face2d, dtype=np.float64)
        face3darr = np.array(face3d, dtype=np.float64)
        smooth_face2darr = np.array(smoothed_face2d, dtype=np.float64)
        smooth_face3darr = np.array(smoothed_face3d, dtype=np.float64)

    cam_matrix = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx],
                           [0, depth_intrinsics.fy, depth_intrinsics.ppy],
                           [0, 0, 1]])
    dist_matrix = np.zeros((5, 1))

    # success, rot_vec, trans_vec = cv2.solvePnP(face3darr, face2darr, cam_matrix, dist_matrix,flags=cv2.SOLVEPNP_ITERATIVE)
    success, rot_vec, trans_vec, inliers = cv2.solvePnPRansac(
        face3darr, face2darr, cam_matrix, dist_matrix,
        iterationsCount=1000, reprojectionError=0.5, confidence=0.99, flags=cv2.SOLVEPNP_DLS)

    rmat, jac = cv2.Rodrigues(rot_vec)
    quat = cv2.RQDecomp3x3(rmat)[3]

    # Calculate Euler angles from quaternion
    yaw_pitch_roll = cv2.RQDecomp3x3(rmat)[0]
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    pitch = yaw_pitch_roll[0] * 360
    yaw = yaw_pitch_roll[1] * 360
    roll = yaw_pitch_roll[2] * 360


    nose3d_projection, jacobian = cv2.projectPoints(smooth_face3darr[1], rot_vec, trans_vec, cam_matrix, dist_matrix)
    # p1 = (int(nose2d[0]), int(nose2d[1]))
    # p2 = (int(nose2d[0] + y * 15), int(nose2d[1] - x * 15))
    p1 = (int(nose3d_projection[0][0][0]), int(nose3d_projection[0][0][1]))
    p2 = (int(smoothed_face2d[1][0] + yaw * 8), int(smoothed_face2d[1][1] - pitch * 8))
    draw_arrowed_line(frame, p1, p2, (255, 255, 255), thickness=2)

    annotated_frame = draw_landmarks_on_image(frame, face_landmarker_result)
    cvzone.putTextRect(annotated_frame, "pitch: " + str(np.round(pitch, 2)), (750, 20), 0.5, cv2.FONT_HERSHEY_PLAIN,
                       (0, 0, 255), 1, cv2.LINE_AA)
    cvzone.putTextRect(annotated_frame, "yaw: " + str(np.round(yaw, 2)), (750, 60), 0.5, cv2.FONT_HERSHEY_PLAIN,
                       (0, 0, 255), 1, cv2.LINE_AA)
    cvzone.putTextRect(annotated_frame, "roll: " + str(np.round(roll, 2)), (750, 100), 0.5, cv2.FONT_HERSHEY_PLAIN,
                       (0, 0, 255), 1, cv2.LINE_AA)



    # # annotated_image = draw_landmarks_on_image(frame, detector.detect_async(mp_image, current_time_ms))
    fps_frame_count += 1
    if (time.time() - fps_start_time) > 1.0:
        fps = fps_frame_count / (time.time() - fps_start_time)
        fps_start_time = time.time()
        fps_frame_count = 0
    cvzone.putTextRect(annotated_frame, 'FPS: {}'.format(int(fps)), (20, 20), 0.5, cv2.FONT_HERSHEY_PLAIN,
                       (0, 0, 255), 1, cv2.LINE_AA)
    image_list = [annotated_frame, depth_image_resized_cm, overlay_image]
    stacked_image = cvzone.stackImages(image_list, 2, 1)
    cv2.imshow('RealsenseView', stacked_image)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

