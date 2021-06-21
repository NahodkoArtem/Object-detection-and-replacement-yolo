import time
import glob
from PIL import Image
import imageio
import cv2
import numpy as np
import pyheal
import sys
import argparse

def createParser():
	parser = argparse.ArgumentParser()
	parser.add_argument('--full', action='store_true', default=False)
	parser.add_argument('--inside', action='store_true', default=False)
	parser.add_argument('--many', action='store_true', default=False)

	return parser

parser = createParser()
namespace = parser.parse_args()
if(namespace.inside==False and namespace.full==False and namespace.many==False):
	print('Не указан параметр замены')
	print('--full для замены на объект исходного размера')
	print('--inside для замены на объект не превосходящий размер исходного')
	print('--many для множественной замены')
	sys.exit()

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
impath = "examples/p0.jpg"
#impath = "examples/p1.jpg"
#impath = "examples/Airplane_2500_m.jpg"
#impath = "examples/Airplane_2500_l.jpg"

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

			mask_img = imageio.imread("mask.png")
			mask = mask_img[:, :, 0].astype(bool, copy=False)
			pyheal.inpaint(image_copy2, mask, 3)
			image_copy2 = cv2.cvtColor(image_copy2, cv2.COLOR_BGR2RGB)
			imageio.imwrite("inpainted.png", image_copy2)

			inpainted = cv2.imread("inpainted.png")
			image_copy2 = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
			imageio.imwrite("inpainted.png", image_copy2)
			src = cv2.imread("helik3.jpg")
			(hel_h, hel_w) = src.shape[:2]

			if namespace.inside:
				hel_h, hel_w = int(hel_h), int(hel_w)
				if hel_w > hel_h:
					koef = float(hel_h / hel_w)
					hel_w = w
					if h < int(hel_w * koef):
						hel_h = h
						hel_w = int(hel_h / koef)
					else:
						hel_h = int(hel_w * koef)
				else:
					koef = float(hel_w / hel_h)
					hel_h = h
					if w < int(hel_h * koef):
						hel_w = w
						hel_h = int(hel_w / koef)
					else:
						hel_w = int(hel_h * koef)
				src_copy = cv2.resize(src, (hel_w, hel_h))
				src = src_copy

			src_mask = np.zeros(src.shape, src.dtype)
			poly = np.array([[0,0], [(hel_w / 2), 0], [hel_w,0], [hel_w, hel_h], [(hel_w / 2),hel_h], [0,hel_h]], np.int32)
			cv2.fillPoly(src_mask, [poly], (255, 255, 255))

			x_centered = int(x + w / 2)
			y_centered = int(y + h / 2)
			if namespace.full or namespace.many:
				inpainted_height = inpainted.shape[0]
				src_height = src.shape[0]
				inpainted_width = inpainted.shape[1]
				src_width = src.shape[1]
				
				if (src_height > inpainted_height or src_width > inpainted_width):
					print("Невозможно произвести замену, размер заменяющего объекта больше размера изображения")
					sys.exit()

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

			output = cv2.seamlessClone(src, inpainted, src_mask, (x_centered, y_centered), cv2.NORMAL_CLONE)

			# Save result
			cv2.imwrite("result.jpg", output)
			if namespace.many:
				image_copy2 = output
			im1 = Image.open('result.jpg')
			im1.show()
		cv2.imshow("image", image_copy)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

detect(impath, net)
