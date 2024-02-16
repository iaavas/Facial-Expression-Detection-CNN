import pygame
import cv2
import numpy as np
import random
import time
import os
from pygame import mixer
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import image


# Load emotion detection model
model = model_from_json(open("Facial Expression Recognition.json", "r").read())
model.load_weights('fer.h5')

# Face detection cascade
face_haar_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')


# Initialize Pygame
pygame.init()
mixer.init()
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Emotion Based Music Player")

# Fonts and colors
font1 = pygame.font.SysFont('JetBrains Mono', 35)

font2 = pygame.font.SysFont('JetBrains Mono', 25)
white = (255, 255, 255)
violet = (50, 205, 50)

black = (0, 0, 0)

play_image = pygame.image.load('./ui/play_button.png')
retake = pygame.image.load('./ui/retake.png')
pause_image = pygame.image.load('./ui/pause_button.png')

button_width = 50
button_height = 50

pause_image = pygame.transform.scale(pause_image, (button_width, button_height))
play_image = pygame.transform.scale(play_image, (50, 50))
retake = pygame.transform.scale(retake, (button_width, button_height))




emotion_start_time = None
emotion_cooldown = 2

# Music variables
current_emotion = None
music_name = None
music_playing = False
music_finished = True

text2 = None


# Function to play music
def play_music(emotion):
    music_folder = f"music/{emotion}"
    music_files = os.listdir(music_folder)
    if music_files and not mixer.music.get_busy():
        
        global music_playing, music_name,music_finished
        
        
        music_finished = False
        
        text2 = font2.render(f"Now Playing for {emotion}:", True, violet)
        
        
        
        music_file = random.choice(music_files)
        music_path = os.path.join(music_folder, music_file)
        mixer.music.load(music_path)
        mixer.music.play()
        music_playing = True
        music_name = music_file.split(".")[0]
        music_finished = True


# Function to draw UI elements
def draw_ui():
    
    text1 = font1.render(f"Detected Emotion: {detected_emotion}", True, black)
    ce = 1
    
    
        
    text3 = font2.render(music_name if music_name else "No music playing", True, white)

    if predictions is not None and len(predictions) > 0:
        confidence_text = font1.render('Confidence: {:.1f}%'.format(np.max(predictions[0])  * 100),True,black)
        screen.blit(confidence_text,(screen_width-250,10))

    screen.blit(text1, (10, 10))
    
    if text2:
        screen.blit(text2, (10, screen_height-150))
    screen.blit(text3, (10, screen_height-110))

    button_pause = pygame.Rect(screen_width / 2, screen_height - 70, 50, 50)
    button_retake = pygame.Rect(screen_width/2 + 100, screen_height - 70, 50, 50)

    if  music_playing == True:
        
        screen.blit(pause_image,(screen_width / 2 , screen_height - 70))
    else:
        screen.blit(play_image, (screen_width / 2 , screen_height - 70))

    screen.blit(retake, (screen_width / 2 + 100 , screen_height - 70))




# Capture video and detect emotions
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()

    # Handle Pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            break

        # Control music playback
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            button_pause = pygame.Rect(screen_width / 2, screen_height - 70, 50, 50)
            button_retake = pygame.Rect(screen_width / 2 + 100, screen_height - 70, 50, 50)
            if button_pause.collidepoint(pos):
                if music_playing :
                    mixer.music.pause()
                    music_playing = False
                    music_finished = False
                else:
                    mixer.music.unpause()
                    music_playing = True
            if button_retake.collidepoint(pos):
                music_finished = True
                mixer.music.stop()

    # Emotion detection on grayscale image
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        # Predict emotions
        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        global detected_emotion
        detected_emotion = emotions[max_index]

        # Check if the detected emotion has changed
        if current_emotion != detected_emotion and  music_finished:
            current_emotion = detected_emotion
            emotion_start_time = time.time()
        elif time.time() - emotion_start_time >= emotion_cooldown and music_finished:
            play_music(detected_emotion)

    # Display the frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    frame = pygame.transform.scale(frame, (screen_width, screen_height))

    screen.fill(black)
    screen.blit(frame, (0, 0))
    draw_ui()
    pygame.display.update()

