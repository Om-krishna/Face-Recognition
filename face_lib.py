import face_recognition
import cv2

video_capture = cv2.VideoCapture(0)

my_image = face_recognition.load_image_file("C:/Users/DELL/Pictures/Camera Roll/WIN_20200126_14_20_47_Pro (2).jpg")
my_face_encoding = face_recognition.face_encodings(my_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    my_face_encoding,
]
known_face_names = [
    "Om Krishna Yadav",
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame=cv2.flip(frame,1)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = cv2.cvtColor(small_frame,cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_names.append(name)
        print(name)
            
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):	
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        print("Top : ",top,"Right : ",right,"Bottom : ",bottom,"Left : ",left)
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    k = cv2.waitKey(1)
    if k== ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
