import pyrealsense2 as rs
import numpy as np
import cvzone
import cv2
import logging


class RealSense:
    def __init__(self, cam_id, res, fps):
        self.cam_id = cam_id
        self.res = res
        self.fps = fps
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.cam_id)
        self.config.enable_stream(rs.stream.depth, self.res[0], self.res[1], rs.format.z16, self.fps)
        self.config.enable_stream(rs.stream.color, self.res[0], self.res[1], rs.format.bgr8, self.fps)
        self.profile = self.pipeline.start(self.config)
        self.sensor = self.profile.get_device().query_sensors()[1]
        for i in range(10):
            self.sensor.set_option(rs.option.enable_auto_exposure, 1)
        # self.sensor.set_option(rs.option.enable_auto_exposure, 1)
        self.hole_filling = rs.hole_filling_filter()
        self.hole_filling.set_option(rs.option.holes_fill, 1)  # Set to mode 1 (Fill from above)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        self.depth_sensor = self.pipeline.get_active_profile().get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        # self.depth_intrinsics = self.profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        # print("Depth Scale is: ", self.depth_scale)
        # Initialize logging for the class
        logging.basicConfig(filename='realsense.log', level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')

    def get_frames(self): # ADD LOGIC FLAG  HERE FLIP TRUE FOR GAZE AND FALSE FOR ARUCO
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                return None, None, None
            color_image = np.asanyarray(color_frame.get_data())
            filled_depth = self.hole_filling.process(depth_frame)
            depth_image = np.asanyarray(filled_depth.get_data())
            # color_image = cv2.flip(color_image, 1)
            # depth_image = cv2.flip(depth_image, 1)

            return depth_frame, depth_image, color_image
        except Exception as e:
            logging.error(f'Error in get_frames(): {e}', exc_info=True)

    def get_framesAruco(self): # ADD LOGIC FLAG  HERE FLIP TRUE FOR GAZE AND FALSE FOR ARUCO

            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame or not color_frame:
                return None, None, None
            color_image = np.asanyarray(color_frame.get_data())
            filled_depth = self.hole_filling.process(depth_frame)
            depth_image = np.asanyarray(filled_depth.get_data())
            return depth_frame, depth_image, color_image


    def get_builtinIntrinsics(self):
        cam_mtx = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                            [0, 0, 1]])
        # cam_mtx_depth = np.array([[self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
        #                           [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
        #                           [0, 0, 1]])

        dist_mtx = np.array(self.color_intrinsics.coeffs)

        return cam_mtx, dist_mtx

    def set_custom_intrinsics(self, K, dist, model):
        # Set  camera intrinsics
        intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics.fx = K[0, 0]
        intrinsics.fy = K[1, 1]
        intrinsics.ppx = K[0, 2]
        intrinsics.ppy = K[1, 2]
        intrinsics.coeffs = dist.flatten().tolist()
        # if using realsense
        # intrinsics.model = model  # Example distortion model
        return intrinsics

    @staticmethod
    def set_custom_extrinsics(R, T):
        extr = rs.extrinsics()
        extr.rotation = R.flatten().tolist()
        extr.translation = T.flatten().tolist()
        return extr

    def stop(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CAM1_ID = '831612073374'
    CAM2_ID = '023422071056'
    CAM3_ID = '827112071528'
    CAM4_ID = '830112071254'
    CAM5_ID = '828612061411'

    res = (640, 480)
    FPS = 30
    cam1 = RealSense(CAM1_ID, res, FPS)
    cam2 = RealSense(CAM2_ID, res, FPS)
    cam3 = RealSense(CAM3_ID, res, FPS)
    cam4 = RealSense(CAM4_ID, res, FPS)
    cam5 = RealSense(CAM5_ID, res,FPS)

    ##############################################
    # Color set
    orange = (14, 94, 255)
    white = (255, 255, 255)
    green = (0, 255, 0)
    black = (0, 0, 0)
    pink = (255, 0, 255)
    teal = (255, 255, 0)
    red = (0, 0, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    ##############################################


    try:
        while True:
            depth_frame1, depth_image1, color_image1 = cam1.get_frames()
            depth_frame2, depth_image2, color_image2 = cam2.get_frames()
            depth_frame3, depth_image3, color_image3 = cam3.get_frames()
            depth_frame4, depth_image4, color_image4 = cam4.get_frames()
            depth_frame5, depth_image5, color_image5 = cam5.get_framesAruco()

            if any(img is None for img in
                   [depth_image1, color_image1, depth_image2, color_image2, depth_image3, color_image3, depth_image4,
                    color_image4,depth_image5,color_image5]):
                continue

            imgList = [color_image1, color_image2, color_image3, color_image4,color_image5]

            cvzone.putTextRect(color_image1, 'CAM1', (5, 30), 0.8, 1, orange, black,
                                                                                    font=font, colorB=white)
            cvzone.putTextRect(color_image2, 'CAM2', (5, 30), 0.8, 1, green, black,
                                                                                    font=font, colorB=white)
            cvzone.putTextRect(color_image3, 'CAM3', (5, 30), 0.8, 1, pink, black,
                                                                                    font=font, colorB=white)
            cvzone.putTextRect(color_image4, 'CAM4', (5, 30), 0.8, 1, red, black,
                                                                                    font=font, colorB=white)
            cvzone.putTextRect(color_image5, 'CAM5', (5, 30), 0.8, 1, teal, black,
                                                                                    font=font, colorB=white)
            stackedImg = cvzone.stackImages(imgList, 3, 0.8)
            cv2.imshow("RealSenseView", stackedImg)
            key = cv2.waitKey(1)
            if key == 27:  # ESC key
                break
    finally:
        cam1.stop()
        cam2.stop()
        cam3.stop()
        cam4.stop()
        cam5.stop()