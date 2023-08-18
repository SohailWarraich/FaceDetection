import cv2
import numpy as np

# s = 0 # Use web camera.
video_cap = cv2.VideoCapture("face.mp4")

win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# Create a network object.
net = cv2.dnn.readNetFromCaffe('deploy.prototxt',
                               'res10_300x300_ssd_iter_140000.caffemodel')

# Model parameters used to train model.
mean = [104, 117, 123]
scale = 1.0
in_width = 300
in_height = 300

# Set the detection threshold for face detections.
detection_threshold = 0.7
pixels = 10

# Annotation settings.
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1

while True:
    ret, frame = video_cap.read()
    if not ret:
        break
    h = frame.shape[0]
    w = frame.shape[1]
    
    # Convert the image into a blob format.
    blob = cv2.dnn.blobFromImage(frame, scalefactor=scale, size=(in_width, in_height), mean=mean, swapRB=False, crop=False)
    
    # Pass the blob to the DNN model.
    net.setInput(blob)
    
    # Retrieve detections from the DNN model.
    detections = net.forward()
    
    # Process each detection.
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > detection_threshold:
            # Extract the bounding box coordinates from the detection.
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype('int')
            roi = frame[y1:y2, x1:x2]
            roi_h, roi_w = roi.shape[:2]
            
    
            if roi_h > pixels and roi_w > pixels:
                # Resize input ROI to the (small) pixelated size.
                roi_small = cv2.resize(roi, (pixels, pixels), interpolation=cv2.INTER_LINEAR)

                # Now enlarge the pixelated ROI to fill the size of the original ROI.
                roi_pixelated = cv2.resize(roi_small, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)
            else:
                roi_pixelated = roi
            
            frame[y1:y2, x1:x2]=roi_pixelated

            # Annotate the video frame with the detection results.
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # label = 'Confidence: %.4f' % confidence
            # label_size, base_line = cv2.getTextSize(label, font_style, font_scale, font_thickness)
            # cv2.rectangle(frame, (x1, y1 - label_size[1]), (x1 + label_size[0], y1 + base_line), (255, 255, 255), cv2.FILLED)
            # cv2.putText(frame, label, (x1, y1), font_style, font_scale, (0, 0, 0), font_thickness)

    cv2.imshow(win_name, frame)
    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
video_cap.release()
cv2.destroyWindow(win_name)