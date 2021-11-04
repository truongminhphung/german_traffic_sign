import numpy as np
import cv2
import tensorflow as tf
import os
from PIL import Image
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Set up paramerters
IMG_HEIGHT = 30
IMG_WIDTH = 30
threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX 

classes = { 0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Vehicle > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }

# Load model
model = tf.keras.models.load_model('model')

# importing libraries
import cv2
import numpy as np
   
# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture('test_video.mp4')
   
# Check if camera opened successfully
if (cap.isOpened()== False): 
  print("Error opening video  file")
   
# Read until video is completed
while(cap.isOpened()):
    start = time.time()
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
      
        image_fromarray = Image.fromarray(frame, 'RGB')
        resize_image = image_fromarray.resize((IMG_HEIGHT, IMG_WIDTH))
        image = np.array(resize_image)
        image = np.expand_dims(image, 0)

        # # Predict image
        pred = model.predict(image)
        probability = np.amax(pred, axis=1)[0]
        class_index = np.argmax(pred,axis=1)[0]
        if probability > threshold:
            cv2.putText(frame, f"Class: {classes[class_index]}", (5, 60), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, f"Probability: {probability*100}%", (5, 130), font, 2, (0, 0, 255), 3, cv2.LINE_AA)

        end = time.time()
        fps = "FPS:" + str(int(1/(end - start)))
        cv2.putText(frame, f"FPS: {str(int(1/(end - start)))}", (5, 190), font, 2, (255, 0, 0), 3, cv2.LINE_AA)
        # Display the resulting frame
        # cv2.imshow('Frame', frame)
        imgS = cv2.resize(frame, (640, 480))
        cv2.imshow('frame', imgS)
    
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
   
  # Break the loop
    else: 
        break
   
# When everything done, release 
# the video capture object
cap.release()
   
# Closes all the frames
cv2.destroyAllWindows()


