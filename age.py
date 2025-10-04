import cv2
from fer import FER
import numpy as np

# Load emotion detector
emotion_detector = FER(mtcnn=True)

# Load age and gender models
age_model = 'models/age_net.caffemodel'
age_proto = 'models/deploy_age.prototxt'
gender_model = 'models/gender_net.caffemodel'
gender_proto = 'models/deploy_gender.prototxt'

age_net = cv2.dnn.readNet(age_model, age_proto)
gender_net = cv2.dnn.readNet(gender_model, gender_proto)

AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = emotion_detector.detect_emotions(frame)

    for face in faces:
        (x, y, w, h) = face["box"]
        face_img = frame[y:y+h, x:x+w].copy()

        # Emotion
        emotion, score = emotion_detector.top_emotion(face_img) or ("Unknown", 0)

        # Age & Gender
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), (78.426, 87.768, 114.895), swapRB=False)
        gender_net.setInput(blob)
        gender = GENDER_LIST[gender_net.forward().argmax()]
        age_net.setInput(blob)
        age = AGE_LIST[age_net.forward().argmax()]

        # Draw box and info
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{gender}, {age}, {emotion}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face + Emotion + Age/Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
