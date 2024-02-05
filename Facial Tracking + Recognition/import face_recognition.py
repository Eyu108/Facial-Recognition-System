import face_recognition
import cv2
import numpy as np
import os
import sys

def load_training_data(training_dir):
    face_encodings = []
    face_names = []

    # Dictionary to map file names to custom names
    name_mapping = {
        'digital photo': 'Name1',
        'digital photo 2': 'Name2'
    }

    if not os.path.exists(training_dir):
        print(f"Training directory {training_dir} not found.")
        return face_encodings, face_names

    for filename in os.listdir(training_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            name_key = os.path.splitext(filename)[0]
            name = name_mapping.get(name_key, "Unknown")

            image_path = os.path.join(training_dir, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                face_encodings.append(encodings[0])
                face_names.append(name)
            else:
                print(f"No faces found in {filename}")
    return face_encodings, face_names

def process_video(video_path, output_path, face_encodings, face_names, frame_skip=5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Unable to open video {video_path}")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings_in_frame = face_recognition.face_encodings(small_frame, face_locations)

            face_names_in_frame = []
            for face_encoding in face_encodings_in_frame:
                matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = face_names[first_match_index]

                face_names_in_frame.append(name)

            for (top, right, bottom, left), name in zip(face_locations, face_names_in_frame):
                top *= 2
                right *= 2
                bottom *= 2
                left *= 2

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        out.write(frame)
        frame_count += 1

    cap.release()
    out.release()

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <video_path> <training_directory>")
        return

    video_path = sys.argv[1]
    training_dir = sys.argv[2]

    known_face_encodings, known_face_names = load_training_data(training_dir)

    if len(known_face_encodings) == 0:
        print("No training data loaded.")
        return

    process_video(video_path, 'output.mp4', known_face_encodings, known_face_names)

if __name__ == "__main__":
    main()
