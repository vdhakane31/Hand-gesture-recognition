import cv2
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageTk
from skimage.feature import hog

IMG_SIZE = 128

model = joblib.load("svm_model.pkl")
gesture_labels = joblib.load("labels.pkl")

display_names = {
    "01_palm": "PALM",
    "03_fist": "FIST",
    "05_thumb": "THUMB UP",
    "10_down": "DOWN"
}

cap = cv2.VideoCapture(0)
gesture_enabled = False

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Hand Gesture Recognition (ML)")
root.geometry("720x620")
root.configure(bg="#1e1e1e")

tk.Label(
    root,
    text="✋ Hand Gesture Recognition",
    font=("Arial", 20, "bold"),
    fg="white",
    bg="#1e1e1e"
).pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

result_label = tk.Label(
    root,
    text="Gesture: ---",
    font=("Arial", 16),
    fg="lime",
    bg="#1e1e1e"
)
result_label.pack(pady=10)

def enable_gesture():
    global gesture_enabled
    gesture_enabled = True
    result_label.config(text="Gesture: Enabled")

def exit_app():
    cap.release()
    root.destroy()

tk.Button(
    root, text="Enable Gesture Control",
    font=("Arial", 14),
    bg="green", fg="white",
    width=25, command=enable_gesture
).pack(pady=5)

tk.Button(
    root, text="Exit App",
    font=("Arial", 14),
    bg="red", fg="white",
    width=25, command=exit_app
).pack(pady=5)

# ---------------- MAIN LOOP ----------------
def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    x1, y1, x2, y2 = 100, 100, 400, 400
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label_text = "Waiting..."

    if gesture_enabled:
        roi = frame[y1:y2, x1:x2]
        roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(16, 16),
            cells_per_block=(2, 2),
            block_norm="L2-Hys"
        )

        probs = model.predict_proba([hog_features])[0]
        idx = np.argmax(probs)
        confidence = probs[idx]
        raw_label = gesture_labels[idx]

        # -------- SMART DECISION --------
        if confidence > 0.10:
            label_text = f"{display_names[raw_label]} ({confidence:.2f})"
        else:
            label_text = "Unknown"

    result_label.config(text=f"Gesture: {label_text}")

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

update_frame()
root.mainloop()
