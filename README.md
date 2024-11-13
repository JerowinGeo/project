# Project

## Aim
To write a python program using OpenCV to do the following image manipulations.
i) Extract ROI from  an image.
ii) Perform handwritting detection in an image.
iii) Perform object detection with label in an image.

## Software Required:
Anaconda - Python 3.7


## Algorithm:
### I)Perform ROI from an image
```python
import cv2
import numpy as np

image = cv2.imread('rose.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.imshow('Original Image', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
roi_mask = np.zeros_like(image_rgb)
roi_mask[100:300, 200:400, :] = 255  
segmented_roi = cv2.bitwise_and(image_rgb, roi_mask)
cv2.imshow('Segmented ROI', segmented_roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
```
#### OUTPUT:
![image](https://github.com/user-attachments/assets/af244a13-88bf-40bc-8087-1cbc7b24d4a0)



### II)Perform handwritting detection in an image

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_handwriting(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 100
    text_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    img_copy = img.copy()
    for contour in text_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
    img_rgb = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title('Handwriting Detection')
    plt.axis('off')
    plt.show()
    
image_path = 'img proj.jpg'
detect_handwriting(image_path)
```
#### OUTPUT:

![image](https://github.com/user-attachments/assets/aad6f3f6-7720-48cf-9b4b-46c806a1de6a)

### III)Perform object detection with label in an image

```python
config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model='frozen_inference_graph.pb'

model=cv2.dnn_DetectionModel(frozen_model,config_file)

classLabels = []
file_name='Labels.txt'
with open(file_name,'rt')as fpt:
    classLabels=fpt.read().rstrip('\n').split('\n')

print(classLabels)
print(len(classLabels))
img=cv2.imread('cow.jpg')
plt.imshow(img)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
model.setInputSize(320,320)
model.setInputScale(1.0/127.5)#255/2=127.5
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale=3
font=cv2.FONT_HERSHEY_PLAIN
for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,0,255),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(255,0,0),thickness=1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
```

#### OUTPUT:
![image](https://github.com/user-attachments/assets/dca122fc-c01d-4c54-8d20-1a37ee1622a5)

## Result:
Thus, a python program using OpenCV for following image manipulations is done successfully.
