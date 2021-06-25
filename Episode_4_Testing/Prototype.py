import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import os


#  Test if camera exists
def test_camera(source):
    capture = cv2.VideoCapture(source)
    camera_available = True
    if capture is None or not capture.isOpened():
        print('Unable to open video camera: ', source)
        camera_available = False

    return camera_available


def person_filter1(output, threshold, height, width):
    class_indexes = []
    confidences_list = []
    boxes_list = []
    centroids_list = []

    for out in output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Person detected
            if (confidence > threshold) and class_id == 0:
                center_x = int(detection[0] * width)
                # from the bounding box to the center
                center_y = int(detection[1] * height)
                # from the bounding box to centroid y
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes_list.append([x, y, w, h])
                confidences_list.append(float(confidence))
                class_indexes.append(class_id)
                centroids_list.append((center_x, center_y))

    return class_indexes, confidences_list, boxes_list, centroids_list


def person_filter2(index, confidence, bounding_box, centroids_list):
    results_list = []
    # Loop over the indexes we are keeping
    for i in index.flatten():
        # Extract the bounding box coordinates
        (x, y) = (bounding_box[i][0], bounding_box[i][1])
        (w, h) = (bounding_box[i][2], bounding_box[i][3])
        # Update our results list to consist of the person prediction probability, bounding box coordinates,
        # and the centroid
        r = (confidence[i], (x, y, x + w, y + h), centroids_list[i])
        results_list.append(r)
    # return the list of results
    return results_list


# BOUNDING BOXES AREA OVERLAPS
def bb_intersection_over_union(box_a, box_b):
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection rectangle
    inter_area = max(0, x_b - x_a + 1) * max(0, y_b - y_a + 1)

    return inter_area


#  DEFINES ####
# Camera list
USB_CAMERA4 = 4
USB_CAMERA3 = 3
USB_CAMERA2 = 2
USB_CAMERA1 = 1
WEB_CAMERA = 0
# Model paths
WEIGHTS = os.getcwd() + "/YOLOV4/yolov4-tiny.weights"
CFG = os.getcwd() + "/YOLOV4/yolov4-tiny.cfg"
COCO = os.getcwd() + "/YOLOV4/coco.names"
# Define the minimum safe distance (in pixels) that two people can be from each other
MIN_DISTANCE = 500
n = 0
THRESHOLD = 0.4
H, W = 480, 640
FONT = cv2.FONT_HERSHEY_PLAIN


# MAIN FUNCTION ####
# Func_net = cv2.dnn.readNet( weights_file, cfg_file)
net = cv2.dnn.readNet(WEIGHTS, CFG)

with open(COCO, "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Uncomment for video streaming
#VIDEO = os.getcwd() + "/Lady_in_the_Red_Dress-Matrix-1999.mp4"
print("[INFO] accessing video stream...")
cap = cv2.VideoCapture(0)


# Test camera ####
# if test_camera(USB_CAMERA4):
#     cap = cv2.VideoCapture(USB_CAMERA4)
# else:
#     if test_camera(USB_CAMERA3):
#         cap = cv2.VideoCapture(USB_CAMERA3)
#     else:
#         if test_camera(USB_CAMERA2):
#             cap = cv2.VideoCapture(USB_CAMERA2)
#         else:
#             if test_camera(USB_CAMERA1):
#                 cap = cv2.VideoCapture(USB_CAMERA1)
#             else:
#                 if test_camera(WEB_CAMERA):
#                     cap = cv2.VideoCapture(WEB_CAMERA)
#                 else:
#                     print("NO AVAILABLE CAMERAS!!")
#                     exit()

# Get frames in real time ####
start_time = time.time()  # Get recording start time
frame_id = 0  # Frame counter

while True:
    hasFrame, frame = cap.read()
    if hasFrame is False:
        print("Frame is empty!!")
        break

    frame = cv2.resize(frame, (W, H), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    frame_id += 1

    # Construct a blob from the input frame
    # Function = blobFromImage(image, scale_factor, size,mean*, swapRB*, crop)
    blob = cv2.dnn.blobFromImage(frame, (1 / 255.0), (320, 320), (0, 0, 0), True, crop=False)
    # and then perform a forward pass of the YOLO object detector, (Standard Yolo Sizes 320x320 and 416x416)
    net.setInput(blob)
    # giving us our bounding boxes and associated probabilities (2 Detections (300, 85) / (1200, 85))
    OUTS = net.forward(output_layers)

    class_ids, confidences, boxes, centroids = person_filter1(OUTS, THRESHOLD, H, W)

    # NON MAX SUPPRESSION (from all bounding boxes taken from and object, filters the one with max area)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.5)
    # Initialize the set of indexes that distance_breach the minimum social distance
    distance_breach = set()

    # Check if indexes not empty
    if len(indexes) > 0:
        # Results consist of(1) the person prediction probability, (2) bounding box coordinates for the detection,
        # and (3) the centroid of the object.
        results = person_filter2(indexes, confidences, boxes, centroids)

        # When 2 or more detections
        if len(results) >= 2:
            # Extract centroids from the results and compute the Euclidean distances between all pairs of the centroids
            detection_boxes = [r[1] for r in results]

            # Loop over the upper triangular of the distance matrix
            for i in range(0, len(detection_boxes)):
                for j in range(i + 1, len(detection_boxes)):
                    # check to see if the distance between any two centroid pairs is less than the configured number
                    # of pixels
                    intersection = bb_intersection_over_union(detection_boxes[i], detection_boxes[j])
                    if intersection > 0:
                        # Update
                        distance_breach.add(i)
                        distance_breach.add(j)

        # Loop over the results
        for (i, (prob, bbox, centroid)) in enumerate(results):
            # extract the bounding box and centroid coordinates, then initialize the color of the annotation
            (startX, startY, endX, endY) = bbox
            (cX, cY) = centroid
            color = (0, 255, 0)

            # if the index pair exists within the violation set, then update the color
            if i in distance_breach:
                color = (0, 0, 255)

            # draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.circle(frame, (cX, cY), 5, (255, 100, 100), -1)

        # draw the total number of social distancing violations on the output frame
        text = "Social distancing breach: {}".format(len(distance_breach))
        cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)

    # Speed tester
    elapse = time.time() - start_time

    fps = frame_id / elapse
    # function_putText = (img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeft)
    cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), FONT, 3, (0, 255, 0), 1)

    # OUTPUT IMAGE
    cv2.imshow("Watchdog", frame) #This line is commented so the program can autostart in the raspberry
    key = cv2.waitKey(1)

    # STOP COMMANDS
    if key == 27:  # esc keyboard
        break

cap.release()  # Close camera reading
cv2.destroyAllWindows()
