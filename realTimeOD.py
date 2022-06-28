import numpy as np
import tensorflow as tf
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import pyaudio
import wave

import cv2

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'data/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# you audio wav here
wf = wave.open('warning.wav', 'rb')

# instantiate PyAudio
p = pyaudio.PyAudio()

# define callback for PyAudio
def callback(in_data, frame_count, time_info, status):
    data = wf.readframes(frame_count)
    return (data, pyaudio.paContinue)

# open stream using callback
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
                stream_callback=callback)

# Stop stream so it wont play the audio first
stream.stop_stream()

# Initialize frame rate calculation
frame_rate_calc = 1
frame_rate_list = []
freq = cv2.getTickFrequency()

# function to load the model
def load_model():

    model_dir = 'exported-model-win-v3\saved_model'

    model = tf.saved_model.load(str(model_dir))

    return model

# function to run the image to the our model
# the output should be a dictionary that contains the
# class object index, detection score, and bounding box
def run_inference_for_single_image(model, image):
    image = np.asarray(image)
    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # Run inference
    model_fn = model.signatures['serving_default']
    output_dict = model_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(output_dict.pop('num_detections'))
    
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    # print('detection_scores :',output_dict['detection_scores'])
    output_dict['num_detections'] = num_detections

    # detection_classes should be ints.
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(
        np.int64)
    # print('detection_classes',output_dict['detection_classes'])

    # Handle models with masks:
    if 'detection_masks' in output_dict:
        # Reframe the the bbox mask to the image size.
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                           tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict

# loading the model object
model = load_model()

# setting the videostream input using opencv
cap = cv2.VideoCapture(0)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# setting the video output for after the program is done
fps_video = int(cap.get(cv2.CAP_PROP_FPS))
writefile = "out-video.mp4"
out_video = cv2.VideoWriter(writefile, cv2.VideoWriter_fourcc(*'avc1'),fps_video,(width,height),True)

# loop while the videostream is open
while cap.isOpened():

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # get frame from video stream
    ret, frame = cap.read()
    image_np = np.array(frame)

    # run the detection function
    detections = run_inference_for_single_image(model, image_np)

    # if there is a weapon detected with a confidance score above 0.3
    # then sound the alarm
    if np.max(detections['detection_scores']) > 0.3:
        if not stream.is_active():
            stream.start_stream()

    # else if there is no weapon detected, then stop the alarm
    elif np.max(detections['detection_scores']) < 0.3:
        if stream.is_active():
            stream.stop_stream()
    
    # make a copy of the image variable
    image_np_with_detections = image_np.copy()

    # draw the bounding box on the image that was copy before
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=5,
        min_score_thresh=.3,
        agnostic_mode=False)
    
    
    # Draw framerate in corner of frame
    cv2.putText(image_np_with_detections,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
    print('FPS: {0:.2f}'.format(frame_rate_calc))
    frame_rate_list.append(frame_rate_calc)

    # show the image in a window
    cv2.imshow('object detection',  image_np_with_detections)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1
    
    # write the frame onto the videostream output
    out_video.write(image_np_with_detections)

    # if the q key on the keyboard is press
    # then stop the script
    if cv2.waitKey(10) & 0xFF == ord('q'):
        if (len(frame_rate_list) != 0):
            print('AVG FPS :',(sum(frame_rate_list)/len(frame_rate_list)))
        cap.release()
        cv2.destroyAllWindows()
        break