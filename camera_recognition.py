import face_recognition
import cv2
import random
import string
import numpy as np
import time

# Constants
WARNING_TIME_LAPSE = 60
WARNING_VIEWS = 1
WARNING_TIME_SINCE_LAST_SEEN = 2
LIMIT_TIME_LAPSE = 120

# Generate some random string to assign as id
def _generate_id(stringLength=8):
    """
    Generate a random string of letters

    :param stringLength: Length of the random string  
    :return: random string
    """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))

def check_face(face_encode, face_record):
    """
    Given one face encoding, check if this face has been seen before, save that record in face_record
    variable and check if this face represents some kind of threat.

    :param face_encode: A face encoding to check
    :param face_record: Variable where all data will be saved (Later will be implemented a kind of DB)
    :return: Tuple (face_record, warning)
    """
    current = time.time()
    listed = False
    warning = False 

    # Iterate over array of faces seen before
    for i in range(len(face_record)):
        
        face = face_record[i]
        encode = face['encode']
        distance = face_recognition.face_distance([face_encode], encode)[0]

        # If the face encode is the same or similar to a face seen before 
        if distance <= 0.6:

            listed = True
            last_seen = face_record[i]['last_time_seen']
            first_seen = face_record[i]['first_time_seen']
            
            if face_record[i]['alerts'] > WARNING_VIEWS:
                warning = True
                
            else:
                # If the face appears IN THE TIME LAPSE and EXCEEDS WARNING_TIME_SINCE_LAST_SEEN
                if current - first_seen < WARNING_TIME_LAPSE and current - last_seen > WARNING_TIME_SINCE_LAST_SEEN:
                    face_record[i]['alerts'] += 1
                    
                # If the last time it was seen exceeds the time limit for the warning 
                elif current - last_seen > LIMIT_TIME_LAPSE:
                    face_record[i]['first_time_seen'] = time.time()
                    warning = False
                
                else:
                    face_record[i]['first_time_seen'] = time.time()

                # If the face represents a threat
                if face_record[i]['alerts'] > WARNING_VIEWS:
                    warning = True

            
            face_record[i]['last_time_seen'] = time.time()
            break
    
    
    # If face has not been seen before
    if listed == False:
        face = {
            'id': _generate_id(),
            'encode': face_encode,
            'first_time_seen': current,
            'last_time_seen': current,
            'alerts': 0
        }
        face_record = [face] + face_record

    return face_record, warning
    

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Initialize some variables

face_record = [] # Contains dictionaries a.k.a. face = {'id': None, 'encode': None, 'time': None}

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
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations, 10, model="large")

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), face_encode in zip(face_locations, face_encodings):

        face_record, warning = check_face(face_encode, face_record)

        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        if warning:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        face = frame[top:bottom, left:right]
        blurred = cv2.blur(face, (50, 50))
        frame[top:bottom, left:right] = blurred
        

    # Display the resulting image
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, "People: " + str(len(face_record)), (0, 20), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()