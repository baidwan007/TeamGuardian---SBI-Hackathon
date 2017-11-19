import face_recognition
import cv2

# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("./sbi_hackathon/sbi1.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces([obama_face_encoding], face_encoding)

        name = "Unknown"
        if match[0]:
            name = "Digvijay"

        # Draw a box around the face
        font = cv2.FONT_HERSHEY_DUPLEX
        if(abs(top-bottom)>=100):
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom +27), (right, bottom), (0, 255, 0), cv2.FILLED)
            cv2.rectangle(frame, (left, top-27), (right, top), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, "LIVE", (left + 6, top- 4), font, 1.0,
                        (255, 255, 255), 1)
        else:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom +27), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.rectangle(frame, (left, top - 27), (right, top), (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "SPOOF", (left + 6, top - 4), font, 1.0,
                        (255, 255, 255), 1)
        # Draw a label with a name below the face
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom + 20), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()