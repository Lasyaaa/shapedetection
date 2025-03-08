from flask import Flask, Response, jsonify
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app, resources={r"/detect": {"origins": "*"}})  # Allow frontend requests

cap = cv2.VideoCapture(0)  # Initialize webcam

def getContours(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detected_shapes = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:  # Ignore small contours
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            sides = len(approx)
            
            # Identify Shape Based on Sides
            if sides == 3:
                shape = "Triangle"
            elif sides == 4:
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = w / float(h)
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif sides > 4:
                shape = "Circle"
            else:
                shape = "Unknown"

            detected_shapes.append(shape)
    
    return detected_shapes

def generate_frames():
    while True:
        success, img = cap.read()
        if not success:
            break  # Stop streaming if the camera fails
        
        # Shape Detection Processing
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        imgCanny = cv2.Canny(imgBlur, 23, 20)
        imgDil = cv2.dilate(imgCanny, np.ones((5, 5)), iterations=1)

        # Get Detected Shapes
        shapes = getContours(imgDil)

        # Draw detected shapes on image
        cv2.putText(img, f"Detected: {', '.join(shapes)}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG format
        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        # Yield frame data for streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect', methods=['GET'])
def detect():
    return jsonify({"message": "Live streaming started at /video_feed"}), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
