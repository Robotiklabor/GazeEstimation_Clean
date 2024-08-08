import pyrealsense2 as rs
import numpy as np
import os
import time
import threading
import cv2
import cvzone


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def initialize_camera(serial_number):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(config)
    sensor = profile.get_device().query_sensors()[1]
    sensor.set_option(rs.option.enable_auto_exposure, 1)
    align = rs.align(rs.stream.color)
    return pipeline, align


def process_frames(pipeline, align):
    try:
        frames = pipeline.wait_for_frames()
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not depth_frame or not color_frame:
        return None, None, None

    color_image = np.asanyarray(color_frame.get_data())

    return color_image


def capture_images_from_camera(serial_number, total_images, delay, folder_name, countdown_time=2):
    pipeline, align = initialize_camera(serial_number)
    create_folder(folder_name)

    start_time = time.time()
    image_count = 0

    try:
        while image_count < total_images:
            color_image = process_frames(pipeline, align)
            if color_image is None:
                continue

            # Calculate elapsed time
            elapsed_time = time.time() - start_time

            # Display countdown before capturing
            if (image_count + 1) * delay - countdown_time <= elapsed_time < (image_count + 1) * delay:
                remaining_time = (image_count + 1) * delay - elapsed_time
                cvzone.putTextRect(color_image, f"Get ready! Capturing in {remaining_time:.1f} sec", (200, 30),
                                   0.6, 1, (0, 255, 0), (0, 0, 0), cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA)

            # Capture and save image at specified intervals
            if elapsed_time >= (image_count + 1) * delay:
                image_count += 1
                image_path = os.path.join(folder_name, f'{image_count}.png')
                cv2.imwrite(image_path, color_image)
                print(f"Image {image_count} saved at {image_path}")

            # Display the capture progress
            cvzone.putTextRect(color_image, f"Image {image_count}/{total_images}", (10, 30),
                               0.6, 1, (0, 0, 255), (0, 0, 0), cv2.FONT_HERSHEY_SIMPLEX, cv2.LINE_AA)
            if folder_name == 'StereoImages/Camera 3':
                cv2.namedWindow(f'Realsense {serial_number}')  # Create window with default properties
                cv2.moveWindow(f'Realsense {serial_number}', 100, 100)  # Set window position
                cv2.imshow(f'Realsense {serial_number}', color_image)
            elif folder_name == 'StereoImages/Camera 2':
                cv2.namedWindow(f'Realsense {serial_number}')  # Create window with default properties
                cv2.moveWindow(f'Realsense {serial_number}', 750, 100)  # Set window position
                cv2.imshow(f'Realsense {serial_number}', color_image)

            # Refresh the frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Release the pipeline
        pipeline.stop()


def main():
    total_images = 12
    delay = 3  # Delay in seconds between captures

    # Serial numbers for each camera
    CAM_3 = '827112071528'
    CAM_2 = '023422071056'
    folder_name_3 = 'StereoImages/Camera 3'
    folder_name_2 = 'StereoImages/Camera 2'

    thread_3 = threading.Thread(target=capture_images_from_camera,
                                args=(CAM_3, total_images, delay, folder_name_3))
    thread_2 = threading.Thread(target=capture_images_from_camera,
                                args=(CAM_2, total_images, delay, folder_name_2))

    thread_3.start()
    thread_2.start()
    thread_3.join()
    thread_2.join()



if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
