# Yolo5 Object Detection and Centroid Finding

**My Environment**
Python: 3.8.12
PyTorch: 1.10.2
OpenCV-Python:  4.5.5.62


In this code I am using [YOLOv5 Algorithm](https://github.com/ultralytics/yolov5) to detect some objects and then finding their **centroids**. The centroids are in pixel coordinates.
This code takes **live camera feed** from a USB webcam and then detect the objects and centroids in **real-time**.
I have trainer the YOLO model using my custom dataset.
To train your custom dataset please follow this guide: [Train Custom Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

**The output of this code is:** ![Result](https://user-images.githubusercontent.com/25352528/153817902-9ae71b55-4bee-4e83-9fe3-d5221052e91f.jpg)
