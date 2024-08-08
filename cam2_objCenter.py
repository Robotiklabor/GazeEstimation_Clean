import os
from RealSenseConfig import RealSense
import numpy as np
import cv2
import cv2.aruco as aruco


class ArucoDetection(RealSense):
    def __init__(self, cam_id, resolution, fps):  # MARKER REAL WRORLD SIZE CM
        super().__init__(cam_id, resolution, fps)
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
        self.parameters = aruco.DetectorParameters()
        # ''''''''''''''  Customise Aruco Detection''''''''''''''''''''''
        self.parameters.minCornerDistanceRate = 0.08
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 50
        self.parameters.cornerRefinementMinAccuracy = 0.1

    @staticmethod
    def transform_and_project(objpoints, rvec, tvec, K, D, *transforms):
        objpoints = objpoints.reshape(-1, 3)
        rvec = rvec.flatten().reshape(3, 1)
        tvec = tvec.flatten().reshape(3, 1)

        for R, T in transforms:
            R = np.array(R, dtype=np.float64)
            T = np.array(T, dtype=np.float64)
            rvec_trans, _ = cv2.Rodrigues(R)
            rvec, tvec = cv2.composeRT(rvec, tvec, rvec_trans, T)[:2]

        rvec = np.array(rvec, dtype=np.float64)
        tvec = np.array(tvec, dtype=np.float64)

        # Project points
        reprojected_points, _ = cv2.projectPoints(objpoints, rvec, tvec, K, D)
        reprojected_points = reprojected_points.reshape(-1, 2)
        reshaped_points_2d = np.expand_dims(reprojected_points, axis=0).reshape(-1, 1, 2)
        undistorted_points_2d = cv2.undistortPoints(reshaped_points_2d, K, D, P=K)

        return np.round(undistorted_points_2d).reshape(-1, 2)

    @staticmethod
    def transform_corner_points(objpoints, rvec, tvec, K, D, *transforms):
        return transform_and_project(objpoints, rvec, tvec, K, D, *transforms)

    @staticmethod
    def transform_center(point3d_src, R, T, cam_mtx_dst, dist_mtx_dst):
        point3d_src = np.array(point3d_src, dtype=np.float64)
        R = np.array(R, dtype=np.float64)
        T = np.array(T, dtype=np.float64)
        cam_mtx_dst = np.array(cam_mtx_dst, dtype=np.float64)
        dist_mtx_dst = np.array(dist_mtx_dst, dtype=np.float64)
        point3d_src_hom = np.append(point3d_src, 1)

        T_matrix = np.eye(4, dtype=np.float64)
        print("Transformation Matrix I:", T_matrix)
        T_matrix[:3, :3] = R
        T_matrix[:3, 3] = T.flatten()
        print("Transformation Matrix:", T_matrix)
        point_transformed_hom = T_matrix @ point3d_src_hom
        point_transformed = point_transformed_hom[:3]
        point_2d_hom = cam_mtx_dst @ point_transformed
        if point_2d_hom[2] != 0:  # Avoid division by zero
            point_2d = point_2d_hom[:2] / point_2d_hom[2]
        else:
            raise ValueError("Projection failed: Z coordinate is zero")
        return point_2d, point_transformed

    def detect_markers(self, color_image, target_id=0):
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (5, 5), 1)
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame_blur, self.marker_dict,
                                                            parameters=self.parameters)
        if marker_IDs is not None:
            valid_corners = []
            valid_ids = []

            for corners, marker_id in zip(marker_corners, marker_IDs):
                if marker_id == target_id:
                    valid_corners.append(corners)
                    valid_ids.append(marker_id)
            return valid_corners, valid_ids
        else:
            return [], []

    def detect_markers_screen(self, color_image, target_id):
        gray_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        gray_frame_blur = cv2.GaussianBlur(gray_frame, (5, 5), 1)
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame_blur, self.marker_dict,
                                                            parameters=self.parameters)

        valid_corners = []
        valid_ids = []

        if marker_IDs is not None:
            for corners, marker_id in zip(marker_corners, marker_IDs.flatten()):  # Flatten to ensure it’s 1D
                if marker_id == target_id:
                    valid_corners.append(corners)
                    valid_ids.append(marker_id)

        return valid_corners, valid_ids

    @staticmethod
    def estimate_pose(marker_corners, marker_IDs, marker_size, color_image, cam_mtx_src, dist_mtx_src):
        global ArucoCenter2d
        if len(marker_corners) == 0:
            print("No markers found")
            return color_image, None  # No valid markers found

        rVec, tVec, object_points = aruco.estimatePoseSingleMarkers(marker_corners, marker_size,
                                                                    cam_mtx_src,
                                                                    dist_mtx_src)

        ArucoCenter3d = tVec
        total_markers = range(0, len(marker_IDs))
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            cv2.polylines(color_image, [corners.astype(np.int32)], True, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.drawFrameAxes(color_image, cam_mtx_src, dist_mtx_src, rVec[i], tVec[i], 20, 2)
            projected_points, _ = cv2.projectPoints(
                tVec[i], np.zeros((3, 1)), np.zeros((3, 1)), cam_mtx_src, dist_mtx_src)
            ArucoCenter2d = projected_points[0].ravel().astype(int)
            cv2.circle(color_image, (int(ArucoCenter2d[0]), int(ArucoCenter2d[1])),
                         5, (255, 0, 255), -1)

        return color_image, ArucoCenter2d, ArucoCenter3d, object_points, rVec, tVec

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


