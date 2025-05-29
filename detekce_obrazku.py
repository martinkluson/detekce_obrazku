import torch
import cv2
import os

MODEL_PATH = "best_test.pt"
IMAGE_FOLDER = "images"
OUTPUT_FOLDER = "output"
CONF_THRESHOLD = 0.5  
IOU_THRESHOLD = 0.5   

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, trust_repo=True, force_reload=True)
model.conf = CONF_THRESHOLD
model.iou = IOU_THRESHOLD

for file_name in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, file_name)
    image = cv2.imread(image_path)
    if image is None:
        print(f"Chyba při načítání: {file_name}")
        continue
    results = model(image, size=960)
    detections = results.xyxy[0]
    for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    cv2.imwrite(output_path, image)
    print(f"{file_name}: Detekováno {len(detections)} rolí.")
