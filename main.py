import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("C:/Users/Selin/OneDrive/Desktop/emotion_model.h5")

emotion_labels = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Sad",
    "Surprise",
    "Neutral"
]

colors = [
    (0,0,255),    # Angry - אדום
    (0,255,0),    # Disgust - ירוק
    (255,0,0),    # Fear - כחול
    (0,255,255),  # Happy - צהוב
    (255,0,255),  # Sad - סגול
    (255,255,0),  # Surprise - תכלת
    (200,200,200) # Neutral - אפור
]

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

N = 10
pred_queue = deque(maxlen=N)

while True:

    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48,48))
        face = face / 255.0
        face = np.reshape(face, (1,48,48,1))

        preds = model.predict(face, verbose=0)[0]

        pred_queue.append(preds)
        avg_preds = np.mean(pred_queue, axis=0)

        label = emotion_labels[np.argmax(avg_preds)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.putText(frame,label,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        bar_x = x
        bar_y = y + h + 20
        bar_height = 15
        for i, (emotion, color) in enumerate(zip(emotion_labels, colors)):
            pct = int(avg_preds[i]*100)
            cv2.rectangle(frame, (bar_x, bar_y + i*(bar_height+5)),
                          (bar_x + pct, bar_y + i*(bar_height+5) + bar_height),
                          color, -1)
            cv2.putText(frame, f"{emotion} {pct}%",
                        (bar_x + 105, bar_y + i*(bar_height+5) + bar_height - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    cv2.imshow("Emotion Camera", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
