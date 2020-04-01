import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open ("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]
# print(layer_names)

# img = cv2.imread("images/sample.jpg")
img = cv2.imread("images/sample2.jpg")
# img = cv2.imread("images/sample.jpg")
# img = cv2.resize(img, None,fx=0.4, fy=0.4)
height, width, channels  = img.shape


blob = cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
# for b in blob:
#     for n, img_blob in enumerate(b):
#         cv2.imshow(str(n),img_blob)

net.setInput(blob)
outs = net.forward(outputlayers)

# print(outs)
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            # cv2.circle(img,(center_x,center_y),10,(255,0,0),2)
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()













