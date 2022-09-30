import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Use OpenCV to get video stream
video_capture = cv2.VideoCapture(0)

# Use face_recognition to load and encode images
dalai_lama_img = face_recognition.load_image_file("photos/Dalai-Lama.jpg")
dalai_lama_encoding = face_recognition.face_encodings(dalai_lama_img)[0]

pope_img = face_recognition.load_image_file("photos/Pope.jpg")
pope_encoding = face_recognition.face_encodings(pope_img)[0]

sam_harris_img = face_recognition.load_image_file("photos/Sam-Harris.jpg")
sam_harris_encoding = face_recognition.face_encodings(sam_harris_img)[0]

# Helpful variables
known_face_encodings = [
    dalai_lama_encoding,
    pope_encoding,
    sam_harris_encoding
]

known_face_names = [
    "Dalai Lama",
    "Pope",
    "Sam Harris"
]

students = known_face_names.copy()
face_locations = []
face_encodings = []
face_names = []
s = True

# Get the current date and open the .csv file
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline = '')
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = ""
            face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_idx = np.argmin(face_distance)
            if matches[best_match_idx]:
                name = known_face_names[best_match_idx]
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    print(f'{name} is here')
                    students.remove(name)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name, current_time])
    cv2.imshow("Attendance system", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
                    