# PPE Detection with YOLOv8

This project demonstrates a computer vision model for **Personal Protective Equipment (PPE) detection**, specifically detecting **helmets, heads, and persons** in images and videos. The model is trained using **YOLOv8**.

---

## 📂 Dataset
The dataset is organized as:
datasets/hardhat/
├── train/
│ └── images, labels
├── valid/
│ └── images, labels
├── test/
│ └── images, labels
└── data.yaml

yaml
- `train`: Training images and labels  
- `valid`: Validation images and labels  
- `test`: Test images for evaluation  
- `data.yaml`: Dataset configuration file

---

## ⚙️ Training
The model was trained in **Google Colab** with the following command:

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")   # Load pretrained YOLOv8n model
results = model.train(
    data="datasets/hardhat/data.yaml",
    epochs=50,
    imgsz=640
)
🎯 Results
Achieved promising results with mAP50 ~ 0.65 on validation data.

Performance can be improved by:

Adding more diverse training images

Increasing epochs

Applying data augmentation

🔍 Inference
Example prediction on a test set:

results = model.predict(source="datasets/hardhat/test", imgsz=640, conf=0.25)
For real-time webcam inference (local machine):

results = model.predict(source=0, show=True)
🚀 Future Work
Improve detection in low light and crowded conditions

Try larger YOLOv8 models (s, m, l) for higher accuracy

Deploy model as a real-time PPE monitoring system

📜 License
This project is for educational purposes only.