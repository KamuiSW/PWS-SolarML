import cv2
import numpy as np
import tensorflow.lite as tflite
import time

IMG_SIZE = (160, 160)
REFERENCE_IMG_PATH = "clean_reference.jpg"
MODEL_PATH = "siamese_clean_detector.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    return img

#loads the refenrece image for clean
ref = cv2.imread(REFERENCE_IMG_PATH)
if ref is None:
    raise RuntimeError("clean_reference.jpg not found")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("cameraless")

print("running stain detection")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    ref_proc = preprocess(ref)
    cur_proc = preprocess(frame)

    interpreter.set_tensor(input_details[0]['index'], ref_proc)
    interpreter.set_tensor(input_details[1]['index'], cur_proc)
    interpreter.invoke()

    score = interpreter.get_tensor(output_details[0]['index'])[0][0]
    dirty = score > 0.5

    text = f"Dirty: {dirty} ({score:.2f})"
    color = (0, 0, 255) if dirty else (0, 255, 0)
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("stain Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
