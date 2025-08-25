import cv2
from ultralytics import YOLO

# Modeli yükle
model = YOLO("best.pt")  # kendi eğittiğin model

# Webcam aç (0 = dahili kamera, 1 harici kamera için)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Model ile tahmin
    results = model.predict(source=frame, conf=0.5, imgsz=640, verbose=False)

    # Çizimli sonuç al
    annotated_frame = results[0].plot()

    # Ekranda göster
    cv2.imshow("YOLOv8 Webcam", annotated_frame)

    # q tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
