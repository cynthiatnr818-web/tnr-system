import sys
import os
import subprocess  
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../..') 
import cv2
import time
import os
from backend.cat_recognition_model.recognition import check_cat_similarity



RTSP_URL = "rtsp://172.20.10.2:554/ch01.264"  #On the left is the address of the video stream corresponding to the teams's specific device.
INTERVAL = 5  
ACTIVITY_THRESHOLD = 1000  
DATABASE_DIR = "/Users/apple/Desktop/TestingDataset"  #On the left is the address of the database on developer's computer. The data is also attached in the appendix.
MODEL_PATH = "/Users/apple/Desktop/research/TNRproject/代码2.0/best_model.pth"  #On the left is the address of the model on developer's computer. The data is also attached in the appendix.
OUTPUT_FOLDER = "/Users/apple/Desktop/research/TNRproject/代码2.0/backend/video_processing/output_frames"  # Folder to save captured frames

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    
def switch_to_relay_wifi():
    subprocess.run(["networksetup", "-setairportnetwork", "en0", "JY_172ab1"]) 
    print("Switched to relay Wi-Fi.")

def switch_to_internet_wifi():
    subprocess.run(["networksetup", "-setairportnetwork", "en0", "FeiFei iPhone (2)", "12345678"])
    print("Switched to internet Wi-Fi.")

def capture_and_detect_activity():
    switch_to_internet_wifi()
    time.sleep(5) 

    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("fail to open, check RTSP")
        return

    print("succeed to open, begin to read frame")

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(frame_rate * INTERVAL)
    frame_count = 0
    last_frame = None
    activity_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("fail to read, check the connection")
            break
        print(f"get frame: {frame_count}")
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video capture.")
            break
        
        if frame_count % frame_interval == 0:
            if last_frame is not None:
                timestamp = int(time.time())
                image_path = os.path.join(OUTPUT_FOLDER, f"frame_{timestamp}.jpg")
                cv2.imwrite(image_path, frame)

                frame_diff = cv2.absdiff(last_frame, frame)
                gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
                _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
                activity_detected = cv2.countNonZero(thresh) > ACTIVITY_THRESHOLD

                if activity_detected:
                    print(f"succeed to recognize activity: {image_path}")
                    activity_frames.append(image_path)
                    
                    if len(activity_frames) == 3:
                        try:
                            print("3 images, call cat-recognition module...")
                            result = check_cat_similarity(
                                image_paths=activity_frames,
                                database_dir=DATABASE_DIR,
                                model_path=MODEL_PATH
                            )
                            

                            
                        except Exception as e:
                            print(f"something went wrong when calling cat-recognition module: {str(e)}")
                        finally:
                            
                            activity_frames = []
                else:

                    activity_frames = []

            last_frame = frame

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    capture_and_detect_activity()