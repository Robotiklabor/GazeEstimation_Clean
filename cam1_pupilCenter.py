import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mediapipe as mp
from RealSenseConfig import RealSense
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyrealsense2 as rs
import numpy as np
import cv2
import time
import cvzone
import logging



class FaceLandmarkDetector:
    def __init__(self, model_path):
        self.result = None
        self.landmarker = self._initialize_landmarker(model_path)

    def _initialize_landmarker(self, model_path):
        options = vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            result_callback=self._print_result)
        return vision.FaceLandmarker.create_from_options(options)

    def _print_result(self, result: mp.tasks.vision.FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result if result.facial_transformation_matrixes else None
        if not self.result:
            print('No face detected.')

    def detect(self, mp_image, timestamp_ms):
        self.landmarker.detect_async(mp_image, timestamp_ms)

    @staticmethod
    def draw_landmarks_on_image(rgb_image, detection_result):
        RED_COLOR = (0, 0, 255)
        GREEN_COLOR = (0, 128, 0)
        ORANGE_COLOR = (0, 165, 255)
        WHITE_COLOR = (224, 224, 224)
        face_landmarks_list = detection_result.face_landmarks
        # annotated_image = np.copy(rgb_image)
        for face_landmarks in face_landmarks_list:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                image=rgb_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=rgb_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=rgb_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())


        return rgb_image

    # @staticmethod
    # def get_3d_point(depth_frame, intrinsics, x, y):
    #     flipped_x = depth_frame.width - x - 1
    #     if 0 <= flipped_x < depth_frame.width and 0 <= y < depth_frame.height:
    #         depth = depth_frame.get_distance(flipped_x, y)
    #         # depth *= 1000  # convert to mm
    #         if depth == 0:
    #             return None
    #
    #         point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [flipped_x, y], depth)
    #         return point_3d
    #     return None

    @staticmethod
    def get_3d_point(depth_frame, intrinsics, x, y):

        if 0 <= x < depth_frame.width and 0 <= y < depth_frame.height:
            depth = depth_frame.get_distance(x, y)
            # depth *= 1000  # convert to mm
            if depth == 0:
                return None

            point_3d = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
            return point_3d
        return None

    @staticmethod
    def project_3d_to_2d(point_3d, intrinsics):  # here color intrinsics should be used
        X, Y, Z = point_3d
        u = (X * intrinsics.fx / Z) + intrinsics.ppx
        v = (Y * intrinsics.fy / Z) + intrinsics.ppy

        return [int(u), int(v)]

    @staticmethod  # project 3dto2d change between color and depth and see the effects
    def get_eyelandmarks(annotated_frame, frame, mask, result, ih, iw, depth_frame, color_intrinsics):
        centerPoint2d, centerPoint3d, centerPoint3d_RS, centerPoint2d_RS = [], [], [], []
        lPupilCen2d, lPupilCen3d, lPupilCen3d_RS, lPupilCen2d_RS = [], [], [], []
        rPupilCen2d, rPupilCen3d, rPupilCen3d_RS, rPupilCen2d_RS = [], [], [], []
        landmarks2d, landmarks3d, landmarks3d_RS, landmarks2d_RS = [], [], [], []
        # LEFT_IRIS = [469, 470, 471, 472]
        # RIGHT_IRIS = [474, 475, 476, 477]
        CENTER_POINT_EYE = [168]
        LEFT_PUPIL_CENTER = [473]
        RIGHT_PUPIL_CENTER = [468]

        for idx, lm in enumerate(result.face_landmarks[0]):
            x, y, z = int(lm.x * iw), int(lm.y * ih), lm.z
            landmarks2d.append([x, y])
            landmarks3d.append([lm.x, lm.y, lm.z])
            point_3d = FaceLandmarkDetector.get_3d_point(depth_frame, color_intrinsics, x, y)

            if idx in CENTER_POINT_EYE:
                if point_3d is not None:
                    centerPoint3d_RS.append(point_3d)
                    centerPoint2d_RS.append(FaceLandmarkDetector.project_3d_to_2d(point_3d, color_intrinsics))


                    centerPoint3d.append([lm.x, lm.y, lm.z])
                    centerPoint2d.append([x, y])
                    # cv2.circle(mask, (x, y), 4, (255, 0, 255), -1)
                    # cv2.circle(mask, (centerPoint2d_RS[0][0], centerPoint2d_RS[0][1]), 2, (255, 255, 0), -1)
                    # cvzone.putTextRect(mask, f"EyeCenter_2D {x, y} ", (5, 30), 0.6, 1,
                    #                    (255, 0, 255), (0, 0, 0),
                    #                    font=cv2.FONT_HERSHEY_SIMPLEX, colorB=(255, 255, 255))
                    # cvzone.putTextRect(mask, f"EyeCenter_2d__RS {centerPoint2d_RS[0][0],centerPoint2d_RS[0][1]} ", (30, 250), 0.6, 1,
                    #                    (255, 225, 0), (0, 0, 0),
                    #                    font=cv2.FONT_HERSHEY_SIMPLEX, colorB=(255, 255, 255))
                    # cvzone.putTextRect(mask,
                    #                    f"EyeCenter_3D {round(point_3d[0], 2), round(point_3d[1], 2), round(point_3d[2], 2)} ",
                    #                    (5, 60), 0.6, 1,
                    #                    (255, 0, 255), (0, 0, 0),
                    #                    font=cv2.FONT_HERSHEY_SIMPLEX, colorB=(255, 255, 255))

            if idx in LEFT_PUPIL_CENTER:
                if point_3d is not None:
                    lPupilCen2d_RS.append(FaceLandmarkDetector.project_3d_to_2d(point_3d, color_intrinsics))
                    lPupilCen3d_RS.append(point_3d)
                lPupilCen3d.append([lm.x, lm.y, lm.z])
                lPupilCen2d.append([x, y])
                # cvzone.putTextRect(mask, f"Left Pupil {x, y} ", (450, 30), 0.5, 1, (0, 255, 0), (0, 0, 0),
                #                    font=cv2.FONT_HERSHEY_SIMPLEX, colorB=(255, 255, 255))

            if idx in RIGHT_PUPIL_CENTER:
                if point_3d is not None:
                    rPupilCen2d_RS.append(FaceLandmarkDetector.project_3d_to_2d(point_3d, color_intrinsics))
                    rPupilCen3d_RS.append(point_3d)
                rPupilCen3d.append([lm.x, lm.y, lm.z])
                rPupilCen2d.append([x, y])
                # cvzone.putTextRect(mask, f"Right Pupil {x, y} ", (20, 30), 0.5, 1, (0, 0, 255), (0, 0, 0),
                #                    font=cv2.FONT_HERSHEY_SIMPLEX, colorB=(255, 255, 255))

            # if idx in LEFT_IRIS + RIGHT_IRIS:
            #     cv2.circle(annotated_frame, (x, y), 2, (0, 255, 0), 1)

        # landmarks2d = np.array(landmarks2d)
        # if len(landmarks2d) >= max(LEFT_IRIS + RIGHT_IRIS):
        #     (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(landmarks2d[LEFT_IRIS])
        #     (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(landmarks2d[RIGHT_IRIS])
        #     center_left = np.array([l_cx, l_cy], dtype=np.int32)
        #     center_right = np.array([r_cx, r_cy], dtype=np.int32)
            # Iris4 Landmarks
            # cv2.circle(frame, center_left, int(l_radius), (0, 0, 255), 1, cv2.LINE_AA)
            # cv2.circle(frame, center_right, int(r_radius), (0, 0, 255), 1, cv2.LINE_AA)
            # Binary Mask
            # cv2.circle(mask, center_left, 2, (255, 255, 255), -1, cv2.LINE_AA)
            # cv2.circle(mask, center_right, 2, (255, 255, 255), -1, cv2.LINE_AA)

        return lPupilCen2d, lPupilCen3d_RS, rPupilCen2d, rPupilCen3d_RS, centerPoint2d, centerPoint3d_RS

    @staticmethod
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


class GazeEstimation(RealSense):
    def __init__(self, model_path, cam_id, resolution, fps):
        super().__init__(cam_id, resolution, fps)
        self.model_path = model_path
        self.detector = FaceLandmarkDetector(model_path)
        self.cam_id = cam_id

    def getFrames(self):
        try:
            while True:
                depthFrame, depthImg, colorImg = self.get_frames()
                # frame = colorImg.copy()
                frame = cv2.cvtColor(colorImg, cv2.COLOR_BGR2RGB)  # why is this needed?
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                frame_timestamp_ms = int(time.time() * 1000)
                self.detector.detect(mp_image, frame_timestamp_ms)
                result = self.detector.result

                if result and result.face_landmarks:
                    annotated_rgb = FaceLandmarkDetector.draw_landmarks_on_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                                                                                 result)
                    # annotated_rgb = colorImg
                    ih, iw, ic = colorImg.shape
                    mask = np.zeros((ih, iw, ic), dtype=np.uint8)
                    # annotated_rgbMask = FaceLandmarkDetector.draw_landmarks_on_image(mask, result, mask=True)
                    lPupilCen2d, lPupilCen3d, rPupilCen2d, rPupilCen3d, centerPoint2d, centerPoint3d = (
                        FaceLandmarkDetector.get_eyelandmarks(
                            annotated_rgb, frame, mask, result, ih, iw, depthFrame,
                            self.color_intrinsics)
                    )
                    eye_list = [lPupilCen2d, lPupilCen3d, rPupilCen2d, rPupilCen3d, centerPoint2d, centerPoint3d]

                    # return depthFrame, depthImg, colorImg, annotated_rgb, annotated_rgbMask, eye_list
                    return depthFrame, depthImg, colorImg, annotated_rgb, eye_list

        except Exception as e:
            logging.error(f'Error in getFrames(): {e}', exc_info=True)


    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    model_path = 'mediapipe_facemodel/face_landmarker.task'
    cam_id = '827112071528'
    resolution = (640, 480)
    fps = 30
    gazeEstimation = GazeEstimation(model_path, cam_id, resolution, fps)

    try:
        while True:
            depthFrame, depthImg, colorImg, annotated_rgb, eye_list = gazeEstimation.getFrames()
            depth_image_3d = np.dstack(
                (depthImg, depthImg, depthImg))
            image_list = [annotated_rgb, annotated_rgbMask]
            stacked_images = cvzone.stackImages(image_list, 2, 1)
            cv2.imshow('GazeEstimation', stacked_images)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    finally:
        gazeEstimation.pipeline.stop()
        cv2.destroyAllWindows()
