# Načteme potřebné knihovny
import torch
import cv2
import os

# Cesty k modelu a složkám s obrázky
MODEL_PATH = "best_test.pt"         # Cesta k natrénovanému YOLOv5 modelu
IMAGE_FOLDER = "images"        # Složka s obrázky
OUTPUT_FOLDER = "output"       # Kam se uloží výsledky

# Vytvoříme složku pro výstupy, pokud ještě neexistuje
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Načteme YOLOv5 model z Ultralytics repozitáře
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, trust_repo=True, force_reload=True)


# Nastavíme prahovou hodnotu důvěry (confidence threshold)
model.conf = 0.3

# Projdeme všechny obrázky ve složce
for file_name in os.listdir(IMAGE_FOLDER):
    image_path = os.path.join(IMAGE_FOLDER, file_name)

    # Načteme obrázek
    image = cv2.imread(image_path)

    # Spustíme detekci
    results = model(image)
    detections = results.xyxy[0]  # [x1, y1, x2, y2, conf, class]

    pocet_roli = len(detections)

    # Vykreslíme jen rámečky bez textu
    for *xyxy, conf, cls in detections.tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Uložíme výsledek
    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    cv2.imwrite(output_path, image)

    # Výpis do konzole
    print(file_name + ": Detekováno " + str(pocet_roli) + " rolí.")
