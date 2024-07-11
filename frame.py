import streamlit as st
from os.path import join, splitext, basename
from ultralytics import YOLO, RTDETR
# import ffmpeg
import cv2
from PIL import Image

# Placeholder for yolov8 inference function
def yolov8_inference(video_path):
    # TODO: Implement yolov8 inference here
    # This function should return the path to the annotated video
    # model = YOLO("yolov8l.pt")
    model =  RTDETR('rtdetr-l.pt')
    results = model(video_path, device=0, conf=0.8, classes=[0,1,2,3,5,7,9,11,12])
    return results[0].plot()

def main():
    st.title("YOLOv8 Object Detection - Frames")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4"])
     # Read video file
    if uploaded_file is not None:

        video = cv2.VideoCapture(uploaded_file.name)
        # fps = int(video.get(cv2.CAP_PROP_FPS))
        fps = 15
        # stframe = st.empty()
        FRAME_WINDOW = st.image([]) 

        frame_count = 0
        while video.isOpened():

            ret, frame = video.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % fps != 0:
                continue  # Skip frames if not equal to FPS

            # Perform inference on the frame
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # stframe.image(img)
            results = yolov8_inference(img)
            # Display the frame with detected objects
            FRAME_WINDOW.image(results, channels="BGR")
            # stframe.image(results, channels="BGR")
            # st.video(results)

        # video.release()

if __name__ == "__main__":
    main()
