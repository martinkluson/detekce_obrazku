# Načteme potřebné knihovny
import torch
import cv2
import os

# Cesty k modelu a složkám s obrázky
MODEL_PATH = "best.pt"         # Cesta k natrénovanému YOLOv5 modelu
IMAGE_FOLDER = "images"        # Složka s obrázky
OUTPUT_FOLDER = "output"       # Kam se uloží výsledky

# Vytvoříme složku pro výstupy, pokud ještě neexistuje
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Načteme YOLOv5 model z Ultralytics repozitáře
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, trust_repo=True)

# Nastavíme prahovou hodnotu důvěry (confidence threshold)
model.conf = 0.3  # Můžeš upravit např. na 0.25 nebo 0.5 dle potřeby

# Projdeme všechny obrázky ve složce
for file_name in os.listdir(IMAGE_FOLDER):
    # Cesta k obrázku
    image_path = os.path.join(IMAGE_FOLDER, file_name)

    # Zpracujeme detekci
    results = model(image_path)

    # Získáme počet detekovaných objektů (rolí)
    detections = results.xyxy[0]  # xyxy formát: [x1, y1, x2, y2, conf, class]
    pocet_roli = len(detections)

    # Vykreslíme obdélníky do obrázku
    obrazek_s_rolimi = results.render()[0]  # Renderovaný obrázek (NumPy array)

    # Uložíme obrázek do výstupní složky
    output_path = os.path.join(OUTPUT_FOLDER, file_name)
    cv2.imwrite(output_path, obrazek_s_rolimi)

    # Výpis výsledku do konzole
    print(file_name + ": Detekováno " + str(pocet_roli) + " rolí.")
