# https://github.com/ayoolaolafenwa/PixelLib

#pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 pixellib opencv-python

#from pixellib.torchbackend.instance import instance_segmentation


import pixellib
from pixellib.instance import instance_segmentation
import cv2

# Initialize the segmentation model
segmentation_model = instance_segmentation()

# Load the model using TensorFlow format
segmentation_model.load_model('/content/mask_rcnn_coco.h5')

# Perform segmentation on the image
segmentation_model.segmentImage("/content/github.jpg",
                                 output_image_name="instance.jpg",
                                 text_size=2, box_thickness=2, text_thickness=2,
                                 show_bboxes=True)


# Video
segmentation_model.process_video("/content/rosewatr.mov", frames_per_second= 15, output_video_name="seg_video.mp4")


# Webcam
capture = cv2.VideoCapture(0)
segmentation_model.process_camera(capture, frames_per_second= 15, output_video_name="output.mp4", show_frames=True,
                              show_bboxes=True,
                              frame_name="frame",
                              extract_segmented_objects=False,
                              save_extracted_objects=False)


""" 
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    
    # Apply instance segmentation
    segmask, output  = segmentation_model.segmentFrame(frame, show_bboxes=True)

    cv2.imshow('Instance Segmentation', output)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()

"""      
