import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
import cvzone
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

def detect_faces(img, model):
    detection = model.predict(img, conf=0.5, verbose=False)
    for info in detection:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            h, w = y2 - y1, x2 - x1
            cvzone.cornerRect(img, [x1, y1, w, h], l=9, rt=3)
            return img
def compute_resultant_vector(landmarks_3d, nose_end_point_3d):
    resultant_vector = np.zeros(3)
    for landmark in landmarks_3d:
        vector = nose_end_point_3d[0] - landmark
        # print(nose_end_point_3d[0])
        resultant_vector += vector
    resultant_vector /= len(landmarks_3d)
    return resultant_vector


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
model_path = r"C:\Users\majid\PycharmProjects\Research Project - Eye Gaze Estimation\facegaze\face_landmarker.task"


options = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    # running_mode=VisionRunningMode.LIVE_STREAM,
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False)

# IntelRealsense
pipe = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.color)
pipe.start(config)
pTime = 0
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
frame_index = 0
detector = vision.FaceLandmarker.create_from_options(options)
kf_list_2d = [KalmanFilter() for _ in range(37)]

while True:
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    # Convert the frame to RGB color space





    ih, iw, ic = frame.shape
    print(frame.shape)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    frame_timestamp_ms = int(time.time() * 1000)
    landmark_list = []
    face2d= []
    face3d= []

    face_landmarker_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
    if not face_landmarker_result.face_landmarks:
        print("No face landmarks detected")
        continue
    for lm in face_landmarker_result.face_landmarks:
        landmark_list.append(lm)
        print(landmark_list[0])
        for idx, landmark in enumerate(landmark_list[0]):
            x, y = landmark.x * iw, landmark.y * ih
            if (idx == 1 or idx == 33 or idx == 463 or idx == 263 or idx == 133
                or idx == 468 or idx == 473 or idx == 78 or idx == 308
                or idx == 0 or idx == 13 or idx == 14 or idx == 17
                or idx == 61 or idx == 291 or idx == 199 or idx == 9
                or idx == 8 or idx == 168 or idx == 6 or idx == 197
                or idx == 195 or idx == 5 or idx == 4 or idx == 13
                or idx == 14 or idx == 17 or idx == 18 or idx == 200
                or idx == 199 or idx == 175 or idx == 152 or idx == 199
                or idx == 55 or idx == 107 or idx == 155 or idx == 285 or idx == 336):
                cv2.circle(frame, (int(x), int(y)), radius=3, color=(0, 255, 0), thickness=-1)


                face2d.append([x, y])
                face3d.append([x, y, landmark.z])


                smoothed_face2d = []
                for i, (x, y) in enumerate(face2d):
                    kf = kf_list_2d[i]
                    smoothed_x, smoothed_y = kf.predict(x, y)
                    cv2.circle(frame, (int(smoothed_x[0]), int(smoothed_y[0])), radius=3, color=(255, 0, 0), thickness=4)
                    smoothed_face2d.append([smoothed_x[0], smoothed_y[0]])

                smoothed_face3d = []
                for i, (x, y, z) in enumerate(face3d):
                    kf = kf_list_2d[i]
                    smoothed_x, smoothed_y = kf.predict(x, y)
                    cv2.circle(frame, (int(smoothed_x[0]), int(smoothed_y[0])), radius=3, color=(255, 0, 0), thickness=4)
                    smoothed_face3d.append([float(smoothed_x[0]), float(smoothed_y[0]), float(z)])
    print(len(face2d))
    face2darr = np.array(face2d, dtype=np.float64)
    face3darr = np.array(face3d, dtype=np.float64)

    smooth_face2darr = np.array(smoothed_face2d, dtype=np.float64)
    smooth_face3darr = np.array(smoothed_face3d, dtype=np.float64)

    focal_length = 1 * iw
    cam_matrix = np.array([[focal_length, 0, ih / 2],
                           [0, focal_length, iw / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec, inliers = cv2.solvePnPRansac(
        face3darr, face2darr, cam_matrix, dist_matrix, flags=cv2.SOLVEPNP_ITERATIVE)

    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    x = angles[0] * 360
    y = angles[1] * 360
    z = angles[2] * 360

    # Define the nose tip point (landmark 1)
    nose_tip_3d = np.array([smooth_face3darr[1]])  # Assuming landmark 1 is at index 0 in your list
    nose_tip_2d = smooth_face2darr[1]

    # Calculate the resultant vector
    resultant_vector = compute_resultant_vector(smooth_face3darr, nose_tip_3d)
    scale_factor = 10  # Adjust this value to make the line longer or shorter
    resultant_vector = resultant_vector * scale_factor
    # Project the resultant vector back to 2D
    resultant_vector_3d = nose_tip_3d + resultant_vector
    resultant_vector_2d, _ = cv2.projectPoints(np.array([resultant_vector_3d]), rot_vec, trans_vec, cam_matrix,
                                               dist_matrix)
    resultant_vector_2d = (int(resultant_vector_2d[0][0][0]), int(resultant_vector_2d[0][0][1]))

    # Draw the resultant vector
    draw_arrowed_line(frame, (int(nose_tip_2d[0]), int(nose_tip_2d[1])), resultant_vector_2d, (0, 255, 0), thickness=3)

    # Optional: Draw individual vectors from landmarks to nose tip for debugging
    for idx, smoothed_3d in enumerate(smooth_face3darr):
        smoothed_2d = smooth_face2darr[idx]
        smoothed_3d_projection, jacobian = cv2.projectPoints(np.array([smoothed_3d]), rot_vec, trans_vec, cam_matrix,
                                                             dist_matrix)
        p1 = (int(smoothed_3d_projection[0][0][0]), int(smoothed_3d_projection[0][0][1]))
        # draw_arrowed_line(frame, p1, (int(nose_tip_2d[0]), int(nose_tip_2d[1])), (255, 255, 255), thickness=1)

    cv2.putText(frame, "x: " + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "y: " + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, "z: " + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    annotated_frame = draw_landmarks_on_image(frame, face_landmarker_result)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 255, 0), 2)
    cv2.imshow('Face Landmark Detection', annotated_frame)
    frame_index += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
