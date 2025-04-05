import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model
import streamlit as st
import webbrowser

# Load the model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Function to detect emotion with live feed display
def detect_emotion_with_display():
    holistic = mp.solutions.holistic
    hands = mp.solutions.hands
    holis = holistic.Holistic()
    drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    detected_emotion = None
    frame_placeholder = st.empty()  # Placeholder in Streamlit for displaying the video feed

    while True:
        lst = []
        ret, frm = cap.read()
        if not ret:
            break

        frm = cv2.flip(frm, 1)
        res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

        # Displaying the frame on Streamlit
        frm_rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frm_rgb, channels="RGB")  # Display the live video frame

        if res.face_landmarks:
            for i in res.face_landmarks.landmark:
                lst.append(i.x - res.face_landmarks.landmark[1].x)
                lst.append(i.y - res.face_landmarks.landmark[1].y)

            if res.left_hand_landmarks:
                for i in res.left_hand_landmarks.landmark:
                    lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            if res.right_hand_landmarks:
                for i in res.right_hand_landmarks.landmark:
                    lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                    lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
            else:
                lst.extend([0.0] * 42)

            lst = np.array(lst).reshape(1, -1)
            pred = label[np.argmax(model.predict(lst))]
            detected_emotion = pred

            # Display the detected emotion on the frame
            cv2.putText(frm, detected_emotion, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Break out of loop after detecting emotion
            break

    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()  # Remove the video feed display after detection
    return detected_emotion

# Streamlit app setup
st.title("Emotion Based Music Recommender")

# Input fields for language and singer (both optional)
language = st.text_input("Enter preferred language")
singer = st.text_input("Enter preferred singer")

# Button to trigger emotion detection and recommendation
if st.button("Recommend me songs"):
    # Detect emotion with live display
    st.write("Opening webcam to detect your emotion. Please wait...")
    detected_emotion = detect_emotion_with_display()

    if detected_emotion:
        # Formulate the YouTube search URL based on inputs
        if language and singer:
            query = f"{language} {detected_emotion} song {singer}"
        elif language:
            query = f"{language} {detected_emotion} song"
        else:
            query = f"{detected_emotion} song"

        youtube_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"

        # Open in a new browser tab
        st.write(f"Opening YouTube with your recommended songs for: {query}...")
        webbrowser.open(youtube_url)
    else:
        st.write("Could not detect emotion. Please try again.")
