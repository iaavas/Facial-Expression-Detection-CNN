import os
import cv2
import numpy as np
import random
import pygame
from pygame import mixer
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image
import time


model = model_from_json(open("Facial Expression Recognition.json", "r").read())
model.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)


pygame.init()
mixer.init()
global music_name

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2
text_color = (255, 255, 255)
text_position = (10, 30)
music_text = None


current_emotion = None
emotion_start_time = None
music_emotion = None


def play_music(emotion):
    music_folder = f"music/{emotion}"
    music_files = os.listdir(music_folder)
    if music_files:
        music_file = random.choice(music_files)
        music_name = music_file
        music_path = os.path.join(music_folder, music_file)
        mixer.music.load(music_path)
        mixer.music.play()
        return music_file
    return None


while cap.isOpened():
    res, frame = cap.read()

    height, width, channel = frame.shape
    
    sub_img = frame[0:50, 0:int(width)]
    csub_img = frame[int(height/1.2):int(height), 0:int(width)]

    black_rect = np.ones(sub_img.shape, dtype=np.uint8) * 0
    cblack_rect = np.ones(csub_img.shape, dtype=np.uint8) * 0
    res = cv2.addWeighted(sub_img, 0.77, black_rect, 0.23, 0)
    cusres = cv2.addWeighted(csub_img, 0.5, cblack_rect, 0.5, 0)
    

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    


    try:
        if music_text!=None:
            cv2.putText(cusres, music_text, (0, 70), font, 0.7, text_color, 2)
            cv2.putText(cusres, f"Detected Emotion: {music_emotion}", (180, 30), font, font_scale, text_color, 2)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, pt1=(x, y), pt2=(x+w, y+h), color=(255, 0, 0), thickness=2)
            roi_gray = gray_image[y-5:y+h+5, x-5:x+w+5]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            image_pixels = img_to_array(roi_gray)
            image_pixels = np.expand_dims(image_pixels, axis=0)
            image_pixels /= 255
            predictions = model.predict(image_pixels)
            max_index = np.argmax(predictions[0])
            emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
            emotion_prediction = emotion_detection[max_index]
            cv2.putText(res, "Sentiment: {}".format(emotion_prediction), (0, 30), font, font_scale, text_color, font_thickness)
            confidence_text = 'Confidence: {:.1f}%'.format(np.max(predictions[0]) * 100)
            cv2.putText(res, confidence_text, (370, 30), font, font_scale, text_color, font_thickness)
            
            
            
            

            
            if current_emotion != emotion_prediction:
                current_emotion = emotion_prediction
                emotion_start_time = time.time()
            elif time.time() - emotion_start_time >= 1 and not mixer.music.get_busy():

                music_folder = f"music/{emotion_prediction}"
                music_files = os.listdir(music_folder)
                if music_files:
                    music_emotion = emotion_prediction
                    music_file = random.choice(music_files)
                    music_text = 'Now Playing: {}'.format(str(music_file))
                    
                    
                    music_path = os.path.join(music_folder, music_file)
                    mixer.music.load(music_path)
                    mixer.music.play()
        

                
    except Exception as e:
        print("Error:", e)

    frame[0:50, 0:int(width)] = res
    frame[int(height/1.2):int(height), 0:int(width)] = cusres
    cv2.imshow('Emotion Detection', frame)

    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
    elif key == ord('r'):
        if mixer.music.get_busy():
            mixer.music.pause()
            time.sleep(1.5)
        else:
            mixer.music.unpause()


cap.release()
cv2.destroyAllWindows()
