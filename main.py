import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
# Assign camera, if only have 1, just take 0
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(
    min_detection_confidence = 0.5) as face_detection:
    while True:
        ret, frame = video_capture.read()
        # If camera not exist stop the app
        if ret == False:
            break
        frame = cv2.flip(frame, 1)
        # convert frame to RGB for detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)
        if results.detections is not None:
            for detection in results.detections:
                # Draw sqare and face points
                mp_drawing.draw_detection(frame, detection,
                mp_drawing.DrawingSpec(color=(250,0,0), circle_radius=1),
                mp_drawing.DrawingSpec(color=(0,252,0)))
        cv2.imshow("Face Scan", frame)
        # Press Q to exit
        if cv2.waitKey(1) == ord('q'):
            break
    
video_capture.release()
cv2.destroyAllWindows()