import os
import cv2
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
import re

def plate(time_formatted, img, img_name):
    return_text  = ""
    # Detect plates in the image using the first YOLO model
    start_time = time.time()
    results_plate = model_plate.predict(img, conf=0.2)[0].boxes.data.tolist()
    return_text = return_text + "model_plate time :"+ f"{(time.time() - start_time) * 1000:.2f}"
    if len(results_plate) == 0 and save:
        # Your code to save the image
        fig = plt.figure(figsize=(16, 9))
        # ...

        print('|No Plate')
        return return_text + "| no plate detected"

    results_plate = sorted(results_plate, key=lambda box: (box[1], box[0]))
    results_plate = sorted(results_plate, key=lambda box: box[0])
    skip_outer_loop = False

    for index_result_plate, result in enumerate(results_plate):
        print('Found Plate ')
        x1, y1, x2, y2, score_plate, class_id = result
        x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
        imgcrop = img[y:y+h, x:x+w]
        plate = ""
        start_time = time.time()
        sorted_results_by_score_plate_ocr = model_plate_ocr.predict(imgcrop)[0].boxes.data.tolist()
        return_text = return_text + "| model_plate_ocr time :"+ f"{(time.time() - start_time) * 1000:.2f}"

        if len(sorted_results_by_score_plate_ocr) <= 3:
            continue

        sorted_results_by_score_plate_ocr = sorted(sorted_results_by_score_plate_ocr, key=lambda box: (box[1], box[0]))
        sorted_results_by_score_plate_ocr = sorted(sorted_results_by_score_plate_ocr, key=lambda box: box[0])

        i = 2

        for index_plate_ocr, result_plate_ocr in enumerate(sorted_results_by_score_plate_ocr):
            i = i + 1
            class_labels = ['0', '1', 'A', 'B', 'waw', 'D', 'H', 'W', 'T', 'CH', 'Maroc', 'J', '2', 'M', '3', '4', '5', '6', '7', '8', '9']
            x1, y1, x2, y2, score_ocr_detection, class_id = result_plate_ocr
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            imgcrop_plate_ocr = imgcrop[y:y+h, x:x+w]
            class_id = class_labels[int(class_id)]

            if score_ocr_detection < 0.7:
                skip_outer_loop = True
                break

            plate += str(class_id)

  
        if skip_outer_loop:
            skip_outer_loop = False
            continue

        pattern_n_c_c = r'^\d+(A|waw|B|D|H|T)\d+$'
        pattern_n_c = r'^\d+(WW|CH|J|Maroc)$'

        if re.match(pattern_n_c_c, plate) or re.match(pattern_n_c, plate):
            return return_text +"| RESULT: "+ plate
        else:
            return return_text + "| pattern"

    return return_text + "| no plate detected"  # Return this value if no plate is found

# Load YOLO models
model_plate = YOLO(os.path.join('.', 'plate', 'best.pt'))
model_plate_ocr = YOLO(os.path.join('.', 'plate-ocr', 'best.pt'))

# Initialize variables
plates_log = [[], []]
rows = 4
cols = 4
save = False  # Set this to True if you want to save images

source_path = "images"  # Replace with the actual source path

log_file = open("processing_times.txt", "w")  # Create a log file to store processing times

for img_name in sorted(os.listdir(source_path), reverse=True):
    start_time = time.time()

    # Load the image from the source path
    img = cv2.imread(os.path.join(source_path, img_name))

    # Process the image with your plate function
    result_plate = plate("", img, img_name)

    elapsed_time_ms = f"{(time.time() - start_time) * 1000:.2f}"

    # Write the image name, processing time, and result to the log file
    log_file.write(f"{img_name}: TOTAL TIME: {elapsed_time_ms} ms | {result_plate}\n")

log_file.close()

# Add any additional code for image processing and result handling here
