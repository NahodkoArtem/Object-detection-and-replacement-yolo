import time
import glob

import imageio
from PIL import Image

import cv2
import numpy as np

import pyheal

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4

impath = "examples/p1.jpg"

weights = glob.glob("yolo/*.weights")[0]
labels = glob.glob("yolo/*.txt")[0]
cfg = glob.glob("yolo/*.cfg")[0]

lbls = list()
with open(labels, "r") as f:
    lbls = [c.strip() for c in f.readlines()]

COLORS = np.random.randint(0, 255, size=(len(lbls), 3), dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weights)

layer = net.getLayerNames()
layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def detect(imgpath, nn):
    image = cv2.imread(imgpath)
    image_copy = image.copy()
    image_copy2 = image.copy()

    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255, (416, 416), swapRB=True, crop=False)
    nn.setInput(blob)
    start_time = time.time()
    layer_outs = nn.forward(layer)
    end_time = time.time()

    boxes = list()
    confidences = list()
    class_ids = list()

    for output in layer_outs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > CONFIDENCE_THRESHOLD:
                if class_id == 4:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (center_x, center_y, width, height) = box.astype("int")

                    x = int(center_x - (width / 2))
                    y = int(center_y - (height / 2))

                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(image_copy, (x, y), (x+w, y+h), color, 2)
            text = "{}: {:.4f}".format(lbls[class_ids[i]], confidences[i])
            cv2.putText(image_copy, text, (x, y -5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            label = "Inference Time: {:.2f} ms".format(end_time - start_time)
            cv2.putText(image_copy, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)
            img = imageio.imread(imgpath)
            mask = cv2.rectangle(img, (x, y), (x+w, y + h), (255, 255, 255), -1)
            mask = cv2.rectangle(img, (0, 0), (W, H - (H - y)), (0, 0, 0), -1)
            mask = cv2.rectangle(img, (0, 0), (W - (W - x), H), (0, 0, 0), -1)
            mask = cv2.rectangle(img, ((x + w), 0), (W, H), (0, 0, 0), -1)
            mask = cv2.rectangle(img, (0, y + h), (W, H), (0, 0, 0), -1)
            cv2.imwrite("mask.png", mask)

            #новый экземпляр чистой картинки
            #img = imageio.imread(imgpath)
            #полученная после дорисовки прямоугольниками маска
            mask_img = imageio.imread("mask.png")
            mask = mask_img[:, :, 0].astype(bool, copy=False)
            pyheal.inpaint(image_copy2, mask, 3)
            #полученный после дорисовки фона результат
            imageio.imwrite("inpainted.png", image_copy2)

            inpainted = cv2.imread("inpainted.png")
            inpainted = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)

            src = cv2.imread("helicopter.png")
            #src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)

            src_mask = np.zeros(src.shape, src.dtype)
            poly = np.array([[29,15], [507,14], [1330,11], [1329,803], [7,803], [33,59]], np.int32)
            cv2.fillPoly(src_mask, [poly], (255, 255, 255))

            x_centered = x + w / 2
            y_centered = y + h / 2

            inpainted_height = inpainted.shape[0]
            src_height = src.shape[0]

            inpainted_width = inpainted.shape[1]
            src_width = src.shape[1]

            if (y_centered - src_height / 2) < 0:
                y = int(y_centered + abs(y_centered - src_height / 2))
            elif (y_centered + src_height / 2) > inpainted_height:
                y = int(y_centered - (y_centered - src_height / 2))
            else:
                y = int(y_centered)

            if (x_centered - src_width / 2) < 0:
                x = int(x_centered + abs(x_centered - src_width / 2))
            elif (x_centered + src_width / 2) > inpainted_width:
                x = int(x_centered - ((x_centered + src_width / 2) - inpainted_width))
            else:
                x = int(x_centered)

            output = cv2.seamlessClone(src, inpainted, src_mask, (x, y), cv2.NORMAL_CLONE)
            #output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)

            # Save result
            cv2.imwrite("result.jpg", output)
            image_copy2 = output
            im1 = Image.open('result.jpg')
            im1.show()
        #cv2.imshow("image", mask)
        #cv2.imshow("image", image_copy)
        #cv2.imwrite("mask.png", mask)
        cv2.imwrite("box.png", image_copy)
        #cv2.waitKey(0)

detect(impath, net)
