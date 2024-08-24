from tkinter import messagebox, Tk, Button, Label, Text, Scrollbar, Canvas
import tkinter
from tkinter.filedialog import askopenfilename
from tkinter import simpledialog
from tkinter import filedialog
from tkinter import ttk
import os
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from PIL import Image, ImageTk
from pygame import mixer

mixer.init()
main = tkinter.Tk()
main.title("RHYTHMIC SOUL")
main.geometry("1920x1080")

# Background Image
bg_image_path = "background_image.jpg"
bg_image = Image.open(bg_image_path)
bg_photo = ImageTk.PhotoImage(bg_image)

# Create Canvas
canvas = Canvas(main, width=1920, height=1080)
canvas.pack(fill="both", expand=True)

# Set Background Image
canvas.create_image(0, 0, image=bg_photo, anchor="nw")

# Fonts
font1 = ('times', 18, 'bold')
font2 = ('times', 14, 'bold')
font3 = ('times', 12, 'bold')

# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(script_dir, 'models')
songs_dir = os.path.join(script_dir, 'songs', 'songsdatabase')

# Variables
value = []
filename = ""
faces = []
frame = None

# Load Models
detection_model_path = os.path.join(models_dir, 'haarcascade_frontalface_default.xml')
emotion_model_path = os.path.join(models_dir, '_mini_XCEPTION.106-0.65.hdf5')
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
predicted_emotion = ""

# Functions
def upload():
    global filename
    global value
    filename = askopenfilename(initialdir="images")
    pathlabel.config(text=filename)

def preprocess():
    global filename
    global frame
    global faces
    text.delete('1.0', tkinter.END)
    orig_frame = cv2.imread(filename)
    orig_frame = cv2.resize(orig_frame, (48, 48))
    frame = cv2.imread(filename, 0)
    faces = face_detection.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    text.insert(tkinter.END, "Total number of faces detected: " + str(len(faces)))

def setPredictedEmotion():
    global faces
    global value
    global predicted_emotion
    if len(faces) > 0:
        faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
        roi = frame[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        preds = emotion_classifier.predict(roi)[0]
        emotion_probability = np.max(preds)
        predicted_emotion = EMOTIONS[preds.argmax()]
        img = cv2.imread(filename)
        img = cv2.resize(img, (400, 400))
        cv2.putText(img, "Emotion Detected As: " + predicted_emotion, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        cv2.imshow("Emotion Detected As: " + predicted_emotion, img)
        cv2.waitKey(0)
        messagebox.showinfo("Emotion Prediction Screen", "Emotion Detected As: " + predicted_emotion)
        path = songs_dir
        value.clear()
        for r, d, f in os.walk(path):
            for file in f:
                if file.find(predicted_emotion) != -1:
                    value.append(file)
        songslist.configure(values=value)
        songslist.current(0)  # Automatically select the first song in the list
    else:
        messagebox.showinfo("Emotion Prediction Screen", "No face detected in the uploaded image")

def playSongFun():
    name = songslist.get()
    if predicted_emotion and name.startswith(predicted_emotion):
        current_song = os.path.join('songs', 'songsdatabase', name)
        mixer.music.load(current_song)
        mixer.music.play()
    else:
        messagebox.showinfo("Song Selection", "Please detect emotion first to select a song.")

def pauseSong():
    mixer.music.pause()

def resumeSong():
    mixer.music.unpause()

def stopSong():
    mixer.music.stop()

# Buttons and Labels
upload_button = Button(main,text="Upload Image With Face", bg='#D3D3D3',command=upload, font=font2)
upload_button.place(x=50, y=100)

pathlabel = Label(main,fg='brown',bg='#D3D3D3', font=font2)
pathlabel.place(x=300, y=100)

preprocess_button = Button(main, text="Preprocess & Detect Face in Image",bg='#D3D3D3', command=preprocess, font=font2)
preprocess_button.place(x=50, y=150)

text = Text(main,fg="brown",bg='#D3D3D3', height=1,width=40,font=font2)
#scroll = Scrollbar(text)
#text.configure(yscrollcommand=scroll.set)
text.place(x=400, y=150)

detect_emotion_button = Button(main,text="Detect Emotion", bg='#D3D3D3',command=setPredictedEmotion, font=font2)
detect_emotion_button.place(x=50, y=200)

emotion_label = Label(main,fg='black', font=font2,bg='#D3D3D3', text="Predicted Song")
emotion_label.place(x=50, y=300)

songslist = ttk.Combobox(main,values=value,postcommand=lambda: songslist.configure(values=value), font=font2)
songslist.place(x=200, y=300)

play_song_button = Button(main, text="Play Song",bg='#D3D3D3', command=playSongFun, font=font2)
play_song_button.place(x=680, y=280)

pause_button = Button(main, fg="orange",text="Pause",bg='#D3D3D3', command=pauseSong, font=font1)
pause_button.place(x=580, y=330)

resume_button = Button(main,fg="green",text="Resume",bg='#D3D3D3', command=resumeSong, font=font1)
resume_button.place(x=680, y=330)

stop_button = Button(main,fg="red",text="Stop",bg='#D3D3D3',command=stopSong, font=font1)
stop_button.place(x=810, y=330)

main.config(bg='brown')
main.mainloop()
