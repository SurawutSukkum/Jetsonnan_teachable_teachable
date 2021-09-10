import cv2
from tensorflow.keras.models import load_model

import numpy as np
import os
import jetson.inference
import jetson.utils 
from datetime import datetime

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
print (DIR_PATH)
directory = f'{DIR_PATH}/NG'

# My constants
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_TYPE = "/dev/video0"


print("START")
sizeTarget = (224, 224)
np.set_printoptions(suppress=True)
dataObj = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

model_path = f'{DIR_PATH}/converted_keras/keras_model.h5'
print(model_path)
model = load_model(model_path) #path model

# Create the camera and display

camera = jetson.utils.videoSource("/dev/video0")

while(True):
     cuda_img  = camera.Capture()
     jetson.utils.cudaDeviceSynchronize()
     print(cuda_img.width)

     cv_img_rgb = jetson.utils.cudaToNumpy(cuda_img)
     cv_img_bgr = cv2.cvtColor(cv_img_rgb, cv2.COLOR_RGB2BGR)

     # Change the current directory 
     # to specified directory 
     os.chdir(directory)
       
     # Filename
     dateTimeObj = datetime.now()
     timestampStr = dateTimeObj.strftime("%d-%b-%Y_%H_%M_%S.jpg")
     filename = timestampStr
       
     # Using cv2.imwrite() method
     # Saving the image
     #cv2.imwrite(filename, cv_img_bgr)
 
     if cv_img_bgr is not None:
        img_resize = cv2.resize(cv_img_bgr,sizeTarget) #resize image      
        image_array = np.asarray(img_resize)#convert image to array
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1 #normalized image

        dataObj[0] = normalized_image_array #get frist dimention 
        prediction =  list(model.predict(dataObj)[0])#change np.ndarray to list 
        idx = prediction.index(max(prediction)) #get index is maximun value

        if  prediction[idx]*100 > 98:
            if idx == 0:
                cv2.putText(cv_img_bgr, "OK: "+str(round(prediction[idx]*100,2)) +"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 255, 100),12,cv2.FILLED)
            elif idx == 2:
                cv2.putText(cv_img_bgr, "No Object: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),12,cv2.FILLED)

        if  prediction[idx]*100 > 50:
            if idx == 1:
                cv2.putText(cv_img_bgr, "NG: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),12,cv2.FILLED)
            elif idx == 2:
                cv2.putText(cv_img_bgr, "No Object: "+str(round(prediction[idx]*100,2))+"%", (50, 100),cv2.FONT_HERSHEY_SIMPLEX, 4,(100, 100, 255),12,cv2.FILLED)


     cv2.imshow("Video Feed", cv_img_bgr)
     c = cv2.waitKey(1)
     if c == 27:
       break

camera.release()
cv2.destroyAllWindows()
