import face_recognition
import cv2
import random
import string
import numpy as np

# Generate some random string to assign as id
def _generate_id(stringLength=8):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def check_face(face_encode):
    listed = False
    for key in face_record:
        encode = face_record[key]
        distance = face_recognition.face_distance([face_encode], encode)[0]
        if distance <= 0.7:
            listed = True
    
    if listed == False:
        face_record[_generate_id()] = face_encode

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables
face_count = {}
face_record = {}

process_this_frame = True
keep_working = True

face_locations = []
face_encodings = []

while keep_working:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame, model="cnn")
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), face_encode in zip(face_locations, face_encodings):

        check_face(face_encode)

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
        face = frame[top:bottom, left:right]
        blurred = cv2.blur(face, (50, 50))
        frame[top:bottom, left:right] = blurred

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, "People: " + str(len(face_record)), (0, 20), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()