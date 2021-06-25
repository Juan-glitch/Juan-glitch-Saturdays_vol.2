# Modules
import serial
from rplidar import RPLidar
import pandas as pd
import time
import cv2
import numpy as np
import time
from scipy.spatial import distance as dist
import os
'''
#$pip install rplidar-roboticia
----------------------------------------------------------------------------------------------------------------
'''
# Defines
angulo_ref = 180
threshold_angle = 20
dist_alpha = 222 #milimetros
dist_beta = 140 #milimetros
dist_gamma = 150 #milimetros
impurity = False
RUN = False
contador = 0
'''
----------------------------------------------------------------------------------------------------------------
'''
# Serial config
stm =serial.Serial('/dev/ttyACM0',115200, bytesize = 8, stopbits = 1,timeout = 0, parity='N')
lidar = RPLidar('/dev/ttyUSB0')

info = lidar.get_info()
print(info)

health = lidar.get_health()
print(health)
'''
----------------------------------------------------------------------------------------------------------------
'''
# YOLO CONFIG

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


'''
----------------------------------------------------------------------------------------------------------------
'''

# MAIN
point = 0
RUN = False
cap = cv2.VideoCapture(0)
# Get frames in real time ####
start_time = time.time()  # Get recording start time
frame_id = 0  # Frame counter

try:
    while(1):
        
        
        # YOLO -------------------------------------------------------------------------------------------------
        hasFrame, frame = cap.read()
        if hasFrame is False:
            print("Frame is empty!!")
            break

        frame = cv2.resize(frame, (W, H), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
        frame_id += 1
        FLAG_IMPURES = False

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
                    FLAG_IMPURES = True

                # draw (1) a bounding box around the person and (2) the centroid coordinates of the person,
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                cv2.circle(frame, (cX, cY), 5, (255, 100, 100), -1)
            
            # draw the total number of social distancing violations on the output frame
            text = "Social distancing breach: {}".format(len(distance_breach))
            cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 3)
        elapse = time.time() - start_time
        fps = frame_id / elapse
        # function_putText = (img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeft)
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 30), FONT, 3, (0, 255, 0), 1)

        # OUTPUT IMAGE
        cv2.imshow("Watchdog", frame) #This line is commented so the program can autostart in the raspberry
        key = cv2.waitKey(1)
        
        # LIDAR -------------------------------------------------------------------------------------------
            
        # Recollect data
        data = pd.DataFrame (columns=['quality', 'angle', 'distance'])
        
        for i, scan in enumerate(lidar.iter_scans()):
            # print(len(scan))
            df_new= pd.DataFrame (scan,columns=['quality', 'angle', 'distance'])

            # if(i > 10):
            data= pd.concat([data, df_new], ignore_index=True)
            if i == 10:
                break
        # Recollect data Lidar
        data.sort_values(by=['angle'], inplace = True)
        maximum =data[['distance']].idxmax()
        calibrate =data[['distance']].idxmin()
        angulo_test = float(data['angle'].iloc[data[['distance']].idxmin()]) # Min distance
        print('el Angulo minimo es: ',angulo_test, calibrate)

        #----------------------------------------------------------------------------------------------------------------
        # Algorithm
        # Cmd 2 send STM
        # Scan
        point = data.iloc[data[['distance']].idxmin()] # Min distance
        scape = data.loc[data[['distance']].idxmax()] # Max distance
        
        # ALRM-SEND
        if((float(point['distance']) < dist_alpha) and (float(point['distance']) > dist_beta) and
           (float(point['angle']) < angulo_ref + threshold_angle) and (float(point['angle']) > angulo_ref - threshold_angle)):
            
            stm.write(b'ALRM-200')
            
        # ALGORITHMIC PARTS ----------------------------------------------------------------------------------------------------------------
        
        # Exception mode
        if(RUN):
            # GAMMA LAYER RUN
            if(  (float(rounout['distance']) > (angulo_ref - threshold_angle)) and 
                 (float(rounout['distance']) < (angulo_ref + threshold_angle))):
                while(rounut != 0):
                    print('continua pa lante gamma') 
                    rounout -=100
                    time.sleep(0.01)
                
            else:
                # Turn LEFT
                if(float(rounout['angle']) - angulo_ref < 0):
                    print('LEFT gamma', abs(float(rounout['angle']) - angulo_ref))
                # Turn RIGHT
                else:
                    print('RIGHT gamma', abs(float(rounout['angle']) - angulo_ref)) 
                    
        # Normal Mode
        else:
            # VOID LAYER
            if(float(point['distance']) >= dist_alpha):
                cmd = 'FORW-200'
                
            # ALPHA LAYER
            elif(float(point['distance']) >= dist_beta):
                # Keep Fwd alpha band
                if(  (float(point['distance']) > (angulo_ref - threshold_angle)) and 
                     (float(point['distance']) < (angulo_ref + threshold_angle))):

                    cmd = 'FORW-200'
                    
                else:
                    # Turn LEFT
                    if(float(point['angle']) - angulo_ref < 0):
                        cmd = 'LEFT-'+ str(abs(float(point['angle']) - angulo_ref))
                    # Turn RIGHT
                    else:
                        cmd = 'RIGH-'+ str(abs(float(point['angle']) - angulo_ref)) 
                        
            # BETA LAYER    
            elif(float(point['distance']) >= dist_gamma):
                    cmd = 'FORW-200' 
            # GAMMA LAYER        
            else:
                # Save Longest distance point
                rounout = scape
                RUN = True
                print('Ejecutar GAMMA')       
        
    # ----------------------------------------------------------------------------------------------------------------------
        # SEND CMDs

        stm.write(cmd.encode())
        while True:

            buffer = stm.read()           # Wait forever for anything
            time.sleep(0.01)              # Sleep (or inWaiting() doesn't give the correct value)
            buffer_left = stm.inWaiting()  # Get the number of characters ready to be read
            
            buffer += stm.read(buffer_left) # Do the read and combine it with the first character
            # INFO: https://stackoverflow.com/questions/13017840/using-pyserial-is-it-possible-to-wait-for-data
            break

        del buffer

        lidar.stop()
        # lidar.stop_motor()
        # lidar.clean_input()

        

except serial.SerialException:
    print('\n*** Serial ports ABRUPTLY closed ***')
    stm.close()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    cap.release()  # Close camera reading
    cv2.destroyAllWindows()
    
    # End functions
except KeyboardInterrupt:
    print('\n*** Serial ports closed ***')
    stm.close()
    lidar.stop()
    lidar.stop_motor()
    lidar.disconnect()
    cap.release()  # Close camera reading
    cv2.destroyAllWindows()

