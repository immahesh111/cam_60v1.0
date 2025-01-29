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

if app_mode == "Live Inspection":
    st.header('Live Mobile Screen Inspection')
    
    # Start video capture using Streamlit's camera input
    run = st.checkbox('Run')
    
    FRAME_WINDOW = st.image([])  # Create an empty image placeholder

    # Create a placeholder for the camera input
    camera_input = st.camera_input("Take a picture")

    if run:
        if camera_input is not None:
            # Convert the uploaded image to an OpenCV format
            file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)

            # Make predictions on the current frame
            result_index = model_prediction(frame)
            
            # Reading Labels
            class_names = ['Cam1_Crack', 'Cam1_FingerPrint', 'Cam1_OK', 'Cam1_Scratch', 'Cam2_FingerPrint', 'Cam2_OK']
            prediction = class_names[result_index]
            
            # Determine color based on prediction
            if prediction in ['Cam1_OK', 'Cam2_OK']:
                color = (0, 255, 0)  # Green for OK
            else:
                color = (0, 0, 255)  # Red for NG (Not Good)

            # Overlay prediction text on frame
            cv2.putText(frame, f'Prediction: {prediction}', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Convert BGR image (OpenCV format) to RGB format (Streamlit format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Display the resulting frame in Streamlit app
            FRAME_WINDOW.image(frame_rgb, channels='RGB')
        else:
            st.warning("Please enable your camera and take a picture.")
