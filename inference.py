import cv2
import numpy as np
import mediapipe as mp
from keras.models import load_model

# Load the trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize Mediapipe Holistic and Drawing tools
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

# Variable to store the detected emotion
detected_emotion = None

while True:
    lst = []

    # Capture frame-by-frame
    ret, frm = cap.read()
    if not ret:
        break
    
    # Flip frame for a mirror effect
    frm = cv2.flip(frm, 1)

    # Process the frame with Mediapipe Holistic
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Check if face landmarks are detected
    if res.face_landmarks:
        # Collect normalized face landmark points
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Collect hand landmark points if hands are detected
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Padding if no left hand detected

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            lst.extend([0.0] * 42)  # Padding if no right hand detected

        # Convert landmarks list to a numpy array and reshape for model input
        lst = np.array(lst).reshape(1, -1)

        # Predict emotion using the model
        pred = label[np.argmax(model.predict(lst))]

        # Display the prediction on the frame
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
        detected_emotion = pred  # Store the detected emotion

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow("Emotion Detection", frm)

    # Break the loop if 'q' is pressed to capture the emotion
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Detected Emotion: {detected_emotion}")
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()

# Print the detected emotion (if needed for further use)
print("Emotion captured:", detected_emotion)
