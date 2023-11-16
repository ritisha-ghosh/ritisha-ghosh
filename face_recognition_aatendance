import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)
# (where you save your code , there create a photo folder and save the students photo into this)
sachin_tendulkar = face_recognition.load_image_file("photo\sachin_tendulkar.jpg")
sachin_tendulkar_encoding = face_recognition.face_encodings(sachin_tendulkar)[0]
sourav_ganguly = face_recognition.load_image_file("photo\sourav_ganguly.jpg")
sourav_ganguly_encoding = face_recognition.face_encodings(sourav_ganguly)[0]
shahrukh_khan = face_recognition.load_image_file("photo\shahrukh_khan.jpg")
shahrukh_khan_encoding = face_recognition.face_encodings(shahrukh_khan)[0]

known_faces_encoding = [
    sachin_tendulkar_encoding,
    sourav_ganguly_encoding,
    shahrukh_khan_encoding
]

known_faces_names =[
    "sachin_tendulkar",
    "sourav_ganguly",
    "shahrukh_khan"
]

students = known_faces_names.copy()

face_locations =[]
face_encodings =[]
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f=open(current_date+'.csv','w+',newline='')
inwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names =[]
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_faces_encoding,face_encoding)
            name =""
            face_distance = face_recognition.face_distance(known_faces_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = s[best_match_index]
        
            face_names.append(name)
            if name in known_faces_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    inwriter.writerow([name,current_time])
                
    cv2.imshow("attendance system",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyALLWindows()
f.close()
