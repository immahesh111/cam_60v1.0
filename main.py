import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# TensorFlow model prediction function
def model_prediction(image):
    model = tf.keras.models.load_model('trained_plant_disease_model_pruned.keras')
    img = cv2.resize(image, (224,224))  # Resize image to match model input
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar configuration for Streamlit appcond
st.sidebar.title('Mobile Camera Inspection')
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "About", "Mobile Inspection", "Live Inspection"])

# Home page
if app_mode == "Home":
    st.header('MOBILE CAMERA INSPECTION SYSTEM')
    image_path = 'home_page.png'
    st.image(image_path, use_column_width=True)
    
    st.markdown('''
                Welcome to the Mobile Screen Inspection System.
                This system is designed to help you inspect your mobile screen whether it is OK or NG.
                Please select the "Mobile Inspection" mode to upload an image of the mobile screen.
                ''')

# About page
elif app_mode == "About":
    st.header("About")
    st.markdown('''
                ### About Dataset
                Go to Mobile Inspection mode to upload an image of the mobile screen. 
                The system will predict if there is a fault in the mobile screen.
                ''')

# Mobile Inspection page
elif app_mode == "Mobile Inspection":
    st.header('Mobile Inspection')
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, width=400, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            # Get prediction result
            result_index = model_prediction(np.array(image))
            
            # Reading Labels
            class_names = ['Cam1_Crack', 'Cam1_FingerPrint', 'Cam1_OK', 'Cam1_Scratch', 'Cam2_FingerPrint', 'Cam2_OK']
            prediction = class_names[result_index]
            
            # Display the prediction
            st.write("Prediction: " + prediction)
            
            # Change background color based on prediction
            if prediction in ['Cam1_OK', 'Cam2_OK']:
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #DFFFD6; /* Light Green */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Mobile Camera is: " + prediction)
            elif prediction in ['Cam1_Scratch', 'Cam1_FingerPrint', 'Cam2_FingerPrint','Cam1_Crack']:
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #FFCCCB; /* Light Red */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.error("Mobile Screen is: " + prediction)
            else:
                # Optional: Handle other predictions if necessary
                st.warning("Prediction does not match expected categories.")



# Assuming model_prediction is defined elsewhere
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import av

# Define your prediction model (assume model_prediction is defined elsewhere)
class VideoProcessor(VideoTransformerBase):
    def __init__(self) -> None:
        self._run = True  # Control processing state

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        if not self._run:
            return frame  # Bypass processing if not running

        # Convert frame to OpenCV format (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Perform prediction on the current frame
        result_index = model_prediction(img)

        # Define class labels
        class_names = ['Cam1_Crack', 'Cam1_FingerPrint', 'Cam1_OK', 
                       'Cam1_Scratch', 'Cam2_FingerPrint', 'Cam2_OK']
        prediction = class_names[result_index]

        # Set text color based on prediction
        color = (0, 255, 0) if prediction in ['Cam1_OK', 'Cam2_OK'] else (0, 0, 255)

        # Overlay prediction text on the frame
        cv2.putText(img, f'Prediction: {prediction}', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Return the annotated frame
        return av.VideoFrame.from_ndarray(img, format="bgr24")

if app_mode == "Live Inspection":
    st.header('Live Mobile Screen Inspection')
    
    # Start/Stop toggle for live processing
    run = st.checkbox('Run', value=True)
    
    # Initialize the video streamer with the custom processor
    ctx = webrtc_streamer(
        key="live-inspection",
        video_processor_factory=VideoProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Control the processing state based on the checkbox
    if ctx.video_processor:
        ctx.video_processor._run = run