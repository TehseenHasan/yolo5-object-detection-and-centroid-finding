import torch
import numpy as np
import cv2
import imutils
import sys
import time
import yaml
import os
import json
import warnings
warnings.filterwarnings("ignore")


# https://github.com/ultralytics/yolov5/issues/6460#issuecomment-1023914166
# https://github.com/ultralytics/yolov5/issues/36


# Loading Model
model = torch.hub.load("yolov5", 'custom', path="yolov5/runs/train/exp/weights/yolo_weights.pt", source='local')  # local repo
#model = torch.hub.load("yolov5", 'custom', path="yolov5/runs/train/exp/weights/yolo_weights.pt", source='local', force_reload=True)  # local repo


# Configuring Model
model.cpu()  #  .cpu() ,or .cuda()
model.conf = 0.25  # NMS confidence threshold
model.iou = 0.45  # NMS IoU threshold
model.agnostic = False  # NMS class-agnostic
model.multi_label = False  # NMS multiple labels per box
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
model.max_det = 20  # maximum number of detections per image
model.amp = False  # Automatic Mixed Precision (AMP) inference




# Function to draw Centroids on the deteted objects and returns updated image
def draw_centroids_on_image(output_image, json_results):   
    data = json.loads(json_results) # Converting JSON array to Python List
    # Accessing each individual object and then getting its xmin, ymin, xmax and ymax to calculate its centroid
    for objects in data:
        xmin = objects["xmin"]
        ymin = objects["ymin"]
        xmax = objects["xmax"]
        ymax = objects["ymax"]
        
        #print("Object: ", data.index(objects))
        #print ("xmin", xmin)
        #print ("ymin", ymin)
        #print ("xmax", xmax)
        #print ("ymax", ymax)
        
        #Centroid Coordinates of detected object
        cx = int((xmin+xmax)/2.0)
        cy = int((ymin+ymax)/2.0)   
        #print(cx,cy)
    
        cv2.circle(output_image, (cx,cy), 2, (0, 0, 255), 2, cv2.FILLED) #draw center dot on detected object
        cv2.putText(output_image, str(str(cx)+" , "+str(cy)), (int(cx)-40, int(cy)+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    return (output_image)
        



if __name__ == "__main__":    
    while(1):
        try:
            #Start reading camera feed (https://answers.opencv.org/question/227535/solvedassertion-error-in-video-capturing/))
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
   
            #Now Place the base_plate_tool on the surface below the camera.
            while(1):
                _,frame = cap.read()
                #frame = undistortImage(frame)
                #cv2.imshow("Live" , frame)
                k = cv2.waitKey(5)
                if k == 27: #exit by pressing Esc key
                    cv2.destroyAllWindows()
                    sys.exit()
                #if k == 13: #execute detection by pressing Enter key           
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # OpenCV image (BGR to RGB)
                
                # Inference
                results = model(image, size=720) #includes NMS

                # Results
                #results.print()  # .print() , .show(), .save(), .crop(), .pandas(), etc.
                #results.show()

                results.xyxy[0]  # im predictions (tensor)
                results.pandas().xyxy[0]  # im predictions (pandas)
                #      xmin    ymin    xmax   ymax  confidence  class    name
                # 0  749.50   43.50  1148.0  704.5    0.874023      0  person
                # 2  114.75  195.75  1095.0  708.0    0.624512      0  person
                # 3  986.00  304.00  1028.0  420.0    0.286865     27     tie
                
                #Results in JSON
                json_results = results.pandas().xyxy[0].to_json(orient="records") # im predictions (JSON)
                #print(json_results)
                
                results.render()  # updates results.imgs with boxes and labels                    
                output_image = results.imgs[0] #output image after rendering
                output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
                
                output_image = draw_centroids_on_image(output_image, json_results) # Draw Centroids on the deteted objects and returns updated image
                
                cv2.imshow("Output", output_image) #Show the output image after rendering
                #cv2.waitKey(1)
                    
                    
                    
        except Exception as e:
            print("Error in Main Loop\n",e)
            cv2.destroyAllWindows()
            sys.exit()
    
    cv2.destroyAllWindows()
    
    
    