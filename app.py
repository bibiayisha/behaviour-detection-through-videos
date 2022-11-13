def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import cv2
from flask import Flask, render_template, Response

from backend.preprocessing import preprocessing
from backend.load_model import load_model

TAG = '[app]'
app = Flask(__name__)
model = load_model()
camera = cv2.VideoCapture(0)
prediction = -1

def generate_frames():
    global prediction
    while True:
        ## read the camera frame
        success, camera_img = camera.read()
        # h, w, c = camera_img.shape
        camera_img = cv2.flip(camera_img, 1)
        # print(TAG, '[camera_img]', camera_img.shape)
        gray = cv2.cvtColor(camera_img, cv2.COLOR_BGR2GRAY)

        if not success:
            break
        else:
            keypoints, record = preprocessing(camera_img)
            # print(TAG, '[record]', type(record), record.shape if record is not None else '')
            # print(record)
            # print(TAG, '[keypoints]', type(keypoints), len(keypoints) if keypoints is not None else '')
            # print(dir(keypoints[0]))
            # camera_img = cv2.drawKeypoints(cv2.resize(gray, (128, 128)), keypoints, cv2.resize(camera_img, (128, 128)))
            # camera_img = cv2.resize(camera_img, (w, h))
            if record is not None and len(record) > 0:
                # record = record[0].reshape(1, -1)
                output = model.predict(record)
                # print(TAG, '[output]', type(output), output)
                count_0 = sum(output == 0)
                count_1 = sum(output == 1)
                prediction = 0 if count_0 > count_1 else 1
                # print(TAG, '[prediction]', prediction)

                ret, buffer = cv2.imencode('.jpg', camera_img)
                # print(TAG, '[buffer]', type(buffer), len(buffer))
                frame = buffer.tobytes()
                # print(TAG, '[frame]', type(frame), len(frame))
            else:
                prediction = -1

        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_result_image():
    global prediction
    while True:
        # print(TAG, '[get_result_image][prediction]', prediction)
        if prediction == 1:
            image_path = 'images/tick.png'
        elif prediction == 0:
            image_path = 'images/cross.png'
        else:
            image_path = 'images/bg-img.jpg'
        # print(TAG, '[image_path]', image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        # print(TAG, '[image]', image.shape)
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        yield(b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/result')
def result():
    return Response(get_result_image(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
