import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model(r'D:\Download\Mask_detection\mask_detection_model.h5')

# Path to the face detection model files
prototxt_path = r'D:\Download\Mask_detection\deploy.prototxt.txt'
caffemodel_path = r'D:\Download\Mask_detection\res10_300x300_ssd_iter_140000.caffemodel'

# Load OpenCV's DNN face detector model
net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)

st.title("Mask Detection in Video Stream")

# Function to predict mask in the face
def predict_mask(face_image, model):
    img = cv2.resize(face_image, (128, 128))  # Ensure the size matches the model's expected input
    img = img / 255.0  # Normalize the image
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return prediction

# Function to run the video stream
def run_video_stream():
    video_capture = cv2.VideoCapture(0)
    stframe = st.empty()
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Prepare the frame for the DNN model
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Adjusted confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                face = frame[y:y1, x:x1]
                if face.shape[0] > 0 and face.shape[1] > 0:
                    prediction = predict_mask(face, model)
                    mask_prob = prediction[0][1]  # Assuming the second label is for "mask"
                    
                    # Draw rectangle around the face
                    label = "Mask" if mask_prob > 0.5 else "No Mask"
                    color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                    cv2.rectangle(frame, (x, y), (x1, y1), color, 2)
                    cv2.putText(frame, f"{label}: {mask_prob:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Display the frame in the Streamlit app
        stframe.image(frame, channels="BGR")

        # Break the loop if "Stop Webcam" button is clicked
        if st.session_state.stop_button_clicked:
            break

    # Release the video capture object
    video_capture.release()

# Initialize session state for button click
if 'stop_button_clicked' not in st.session_state:
    st.session_state.stop_button_clicked = False

# Button to start webcam
if st.button("Start Webcam"):
    st.session_state.stop_button_clicked = False
    run_video_stream()

# Button to stop webcam
if st.button("Stop Webcam"):
    st.session_state.stop_button_clicked = True

# Upload image for mobile camera
uploaded_file = st.file_uploader("Upload a photo from your mobile camera", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = np.array(image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the image for the DNN model
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:  # Adjusted confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            face = image[y:y1, x:x1]
            if face.shape[0] > 0 and face.shape[1] > 0:
                prediction = predict_mask(face, model)
                mask_prob = prediction[0][1]  # Assuming the second label is for "mask"
                
                # Draw rectangle around the face
                label = "Mask" if mask_prob > 0.5 else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (255, 0, 0)
                cv2.rectangle(image, (x, y), (x1, y1), color, 2)
                cv2.putText(image, f"{label}: {mask_prob:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    
    # Display the result image
    st.image(image, caption='Processed Image', use_column_width=True)
