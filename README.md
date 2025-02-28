# 🥤 Real-Time Bottle Detection with YOLO | AI-Powered Object Detection

## 🚀 Overview

This project utilizes **YOLO (You Only Look Once)**, a state-of-the-art object detection model, to detect bottles in real-time from video streams. Using **OpenCV** and **YOLO**, the system processes frames, identifies bottles, and highlights them with bounding boxes.

This technology has practical applications in **automation, quality control, and inventory management**.

---

## 🔗 Features

✅ **Real-time Bottle Detection** using **YOLO**
✅ **Frame-by-frame Analysis** with OpenCV
✅ **Confidence-based Filtering** to reduce false positives
✅ **Bounding Box Visualization** for detected bottles
✅ **Video Processing and Output Saving**
✅ **Scalable for Industrial Applications** (e.g., assembly line bottle counting)

---

## 🛠️ Technologies Used

🔹 **Python** - Programming language for AI development  
🔹 **OpenCV** - Image processing and computer vision library  
🔹 **YOLO (You Only Look Once)** - Object detection model  
🔹 **NumPy** - Used for efficient numerical computations  
🔹 **Ultralytics YOLO** - Pretrained model for object detection  

---

## 📸 Demo Preview

[![Real-Time Bottle Detection](https://youtube.com/shorts/cjj7zAqm290?feature=share)](https://youtube.com/shorts/cjj7zAqm290?feature=share)

---

## 📝 Installation & Setup

### **1️⃣ Clone the Repository**

```bash
git clone https://github.com/Armanlaliwala/Bottle-Detection-YOLO
cd Bottle-Detection-YOLO
```

### **2️⃣ Install Dependencies**

Ensure you have Python installed, then run:

```bash
pip install opencv-python ultralytics numpy
```

### **3️⃣ Run the Script**

```bash
python bottle_detection.py
```

### **4️⃣ Controls**

🔹 Press **'Q'** to stop the video

---

## 🔍 How It Works

1️⃣ **Loads the YOLO model** pretrained on the COCO dataset.  
2️⃣ **Reads frames from a video file** using OpenCV.  
3️⃣ **Runs YOLO detection** on each frame to identify bottles.  
4️⃣ **Filters detections** based on the COCO class ID for bottles (class 39).  
5️⃣ **Draws bounding boxes** around detected bottles.  
6️⃣ **Displays the processed frame** in real-time.  
7️⃣ **Saves the output** as a new video file.  

---

## 🌟 Enhancements & Future Scope

🔹 **Optimize YOLO model** for higher accuracy with custom training.  
🔹 **Integrate real-time video streaming** (e.g., CCTV for inventory tracking).  
🔹 **Expand detection categories** (e.g., detecting different types of bottles).  
🔹 **Deploy as a web app** with Flask or FastAPI.  

---

## 🤝 Contributing

Contributions are welcome! Feel free to **fork** the repository, make improvements, and submit a **pull request**.

---

## 💌 Contact & Support

For any issues or suggestions, feel free to **open an issue** or contact me:  
👨‍💻 **Arman Laliwala**  
💎 [LinkedIn](https://www.linkedin.com/in/armanlaliwala/)  
📧 **Email:** [armanlaliwala@gmail.com](mailto:armanlaliwala@gmail.com)  

---

🌟 **If you found this project helpful, don’t forget to give it a star!** ⭐

---

## 📚 Code Explanation

```python
import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load video
video_path = '2.mp4'
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
output_path = 'output_beer_bottle_detection.mp4'
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

print("Processing video...")

# Process video frame by frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO detection
    results = model.predict(frame, conf=0.5)

    # Filter detections for bottles (COCO class 39)
    bottle_detections = [det for det in results[0].boxes if int(det.cls) == 39]

    # Draw bounding boxes
    for det in bottle_detections:
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        confidence = det.conf[0]
        label = f"Bottle {confidence:.2f}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('Bottle Detection', frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Detection completed. Output saved to '{output_path}'")
```

### **Code Breakdown**

1️⃣ **Load YOLO Model** - The model is loaded with a pre-trained COCO dataset.  
2️⃣ **Read Frames from Video** - OpenCV captures each frame for processing.  
3️⃣ **Run YOLO Detection** - YOLO analyzes the frame and detects objects.  
4️⃣ **Filter for Bottles** - Only objects classified as 'bottle' (class 39) are considered.  
5️⃣ **Draw Bounding Boxes** - A green box highlights detected bottles.  
6️⃣ **Display & Save Output** - The processed frames are displayed and stored as a new video.  

---

🌟 **Hope this README helps! Let me know if you have any suggestions!** 🚀

