import cv2
import tensorflow as tf

def bound(boxes, scores, h, w):
    idxs = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 1.5)

    # define a array as matrix
    signs = []
    for i in range(len(idxs)):
            signs.append(i)
    height, width = h, w
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            ymin = int((boxes[i][0] * height))
            xmin = int((boxes[i][1] * width))
            ymax = int((boxes[i][2] * height))
            xmax = int((boxes[i][3] * width))
            signs[i] = [ymin,ymax,xmin,xmax]
    return signs

def draw_bounding_box(frame, detect_fn):
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    boxes = detections['detection_boxes']
    scores = detections['detection_scores']
    h, w = frame.shape[:2]
    boxes = boxes.tolist()
    scores = scores.tolist()
    coordinates = bound(boxes, scores, h, w)
    return coordinates
