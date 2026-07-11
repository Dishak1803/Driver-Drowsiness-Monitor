import cv2
import numpy as np
import threading
import os
from tensorflow.keras.models import load_model

print("Loading models...")
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
eye_model  = load_model(os.path.join(BASE_DIR, "eye_model.h5"))
yawn_model = load_model(os.path.join(BASE_DIR, "yawn_model.h5"))
print("✅ Models loaded!")

face_cascade  = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade   = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

EYE_CLOSED_FRAMES  = 10
EYE_THRESHOLD  = 0.3
YAWN_THRESHOLD = 0.4
closed_frame_count = 0
alarm_playing      = False

def play_alarm():
    global alarm_playing
    alarm_playing = True
    for _ in range(5):
        if not alarm_playing:
            break
        os.system('powershell -c "[console]::beep(2500,300)"')
        os.system('powershell -c "[console]::beep(2000,300)"')
    alarm_playing = False

def trigger_alarm():
    global alarm_playing
    if not alarm_playing:
        t = threading.Thread(target=play_alarm, daemon=True)
        t.start()

def preprocess(img):
    img = cv2.convertScaleAbs(img, alpha=0.6, beta=-30)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def draw_status(frame, text, color):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h-50), (w, h), color, -1)
    cv2.putText(frame, text, (10, h-15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam!")
    exit()

print("✅ Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eye_status  = "Eye: --"
    yawn_status = "Yawn: --"
    is_drowsy   = False

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(80,80))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

        face_color = frame[y:y+h, x:x+w]
        face_gray  = gray[y:y+h, x:x+w]

        # ── Eye Detection ──────────────────────────────────────
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20,20))

        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]
            cv2.rectangle(face_color, (ex,ey), (ex+ew,ey+eh), (0,255,0), 2)
            eye_crop = face_color[ey:ey+eh, ex:ex+ew]
            eye_pred = eye_model.predict(preprocess(eye_crop), verbose=0)[0][0]

            if eye_pred > EYE_THRESHOLD:
                eye_status = "Eye: Open"
                closed_frame_count = 0
            else:
                eye_status = "Eye: Closed"
                closed_frame_count += 1
        else:
            closed_frame_count += 1
            eye_status = "Eye: Not Detected"

        # ── Mouth Detection ────────────────────────────────────
        mouths = mouth_cascade.detectMultiScale(face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25,25))

        if len(mouths) > 0:
            (mx, my, mw, mh) = mouths[0]
            cv2.rectangle(face_color, (mx,my), (mx+mw,my+mh), (0,165,255), 2)
            mouth_crop = face_color[my:my+mh, mx:mx+mw]
            yawn_pred  = yawn_model.predict(preprocess(mouth_crop), verbose=0)[0][0]

            if yawn_pred > YAWN_THRESHOLD:
                yawn_status = "Yawn: YES"
                is_drowsy   = True
            else:
                yawn_status = "Yawn: No"

        if closed_frame_count >= EYE_CLOSED_FRAMES:
            is_drowsy = True

        cv2.putText(frame, eye_status,  (x, y-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.putText(frame, yawn_status, (x, y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        cv2.putText(frame, f"Closed Frames: {closed_frame_count}/{EYE_CLOSED_FRAMES}",
                    (x, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

    # ── Alert ──────────────────────────────────────────────────
    if is_drowsy:
        draw_status(frame, "WARNING: DROWSINESS DETECTED - TAKE A BREAK!", (0,0,200))
        trigger_alarm()
    elif len(faces) > 0:
        draw_status(frame, "ALERT - Drive Safe!", (0,150,0))
        alarm_playing = False
    else:
        draw_status(frame, "No Face Detected", (80,80,80))
        alarm_playing = False

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        alarm_playing = False
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done!")