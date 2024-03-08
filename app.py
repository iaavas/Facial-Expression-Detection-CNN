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

model = model_from_json(open("Facial Expression Recognition.json", "r").read())
model.load_weights('fer.h5')

face_haar_cascade = cv2.CascadeClassifier('static\haarcascade_frontalface_default.xml')

pygame.init()
mixer.init()

# Screen settings for emotion detection
screen_width = 600
screen_height = 600
screen = pygame.display.set_mode((screen_width * 2, screen_height))
pygame.display.set_caption("Emotion Based Music Recommendation")

# Screen settings for displaying recommended music
music_screen_width = 300
music_screen_height = 600
music_screen = pygame.Surface((music_screen_width, music_screen_height))
pygame.display.set_caption("Recommended Music")

font1 = pygame.font.SysFont('FreeSansBold.ttf', 35)
font2 = pygame.font.SysFont('FreeSansBold.ttf', 25)
font3 = pygame.font.SysFont('FreeSansBold.ttf', 30)  
white = (255, 255, 255)
violet = (50, 255, 50)
black = (0, 0, 0)
gray = (128, 128, 128)  # New color for dimmed text

retake = pygame.image.load('./ui/retake.png')
button_width = 50
button_height = 50
retake = pygame.transform.scale(retake, (button_width, button_height))

emotion_start_time = 0
emotion_cooldown = 10

current_emotion = None
music_recommendations = []
music_emotion = ''
retake_clicked_time = 0
text2 = None

consecutive_emotion_count = 0
consecutive_emotion_threshold = 3  

def recommend_music(emotion):
    global music_recommendations, music_emotion
    music_folder = f"music/{emotion}"
    music_files = os.listdir(music_folder)
    if music_files:
        music_emotion = emotion
        music_recommendations = random.sample(music_files,7)
        music_recommendations = [music_file.split(".")[0] for music_file in music_recommendations]
    else:
        music_recommendations = []

def draw_music_screen():
    music_screen.fill(black)
    if music_recommendations:
        for i, music in enumerate(music_recommendations):
            text = font2.render(music, True, white)
            music_screen.blit(text, (20, 20 + i * 40))  # Increased line spacing
    else:
        text = font2.render("No recommendation available", True, violet)
        music_screen.blit(text, (20, 20))

    # Draw a background for the music details
    music_details_bg = pygame.Surface((music_screen_width - 40, 120), pygame.SRCALPHA)
    music_details_bg.fill((0, 0, 0, 200))  #
    music_screen.blit(music_details_bg, (20, music_screen_height - 140))

    # Draw the music emotion and recommendation count
    if music_recommendations:
        emotion_text = font3.render(f"Emotion: {music_emotion.capitalize()}", True, white)
        count_text = font3.render(f"Recommendations: {len(music_recommendations)}", True, white)
        music_screen.blit(emotion_text, (30, music_screen_height - 130))
        music_screen.blit(count_text, (30, music_screen_height - 110))
    else:
        dimmed_text = font3.render("No recommendations available", True, gray)
        music_screen.blit(dimmed_text, (30, music_screen_height - 120))

    screen.blit(music_screen, (screen_width, 0))
    pygame.display.update()

def draw_ui():
    text1 = font1.render(f"Detected Emotion: {detected_emotion.capitalize()}", True, white)

    if predictions is not None and len(predictions) > 0:
        confidence_text = font1.render('Confidence: {:.1f}%'.format(np.max(predictions[0]) * 100), True, white)
        screen.blit(confidence_text, (screen_width - 250, 10))

    overlay = pygame.Surface((screen_width, 150), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 128))  # Black with alpha value (128 for semi-transparent)
    # screen.blit(overlay, (0, screen_height - 120))

    screen.blit(text1, (10, 10))

cap = cv2.VideoCapture(0)
detected_emotion = ""
predictions = []
while True:
    current_time = time.time()
    ret, frame = cap.read()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            cv2.destroyAllWindows()
            pygame.quit()
            break

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            button_retake = pygame.Rect(screen_width / 2 + 100, screen_height - 60, 50, 50)
            if button_retake.collidepoint(pos):
                emotion_start_time = time.time()
                retake_clicked_time = current_time
                music_recommendations = []
                consecutive_emotion_count = 0  # Reset consecutive emotion count

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_haar_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray_image[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)
        max_index = np.argmax(predictions[0])
        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        new_detected_emotion = emotions[max_index]

        if new_detected_emotion == detected_emotion:
            consecutive_emotion_count += 1
        else:
            consecutive_emotion_count = 1

        detected_emotion = new_detected_emotion

        if consecutive_emotion_count >= consecutive_emotion_threshold:
            if current_emotion != detected_emotion:
                current_emotion = detected_emotion
                recommend_music(detected_emotion)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    frame = pygame.transform.scale(frame, (screen_width, screen_height))

    screen.fill(black)
    screen.blit(frame, (0, 0))
    draw_ui()
    draw_music_screen()

    pygame.display.update()