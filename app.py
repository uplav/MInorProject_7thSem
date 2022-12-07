from flask import Flask, render_template, request,Response
import cv2 ,time
from tensorflow.keras.models import load_model
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from pytz import timezone
import jyserver.Flask as jsf

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///pose_estimate.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)
        
        

camera=cv2.VideoCapture(0)
new_model=load_model(r'C:\Users\UPLAV DANG\Desktop\python_webdev\Final_Project\yoga_6model.h5')
captured_frames=[]

class exercise(db.Model):
    exercise_id=db.Column(db.Integer, primary_key=True)
    duration=db.Column(db.Integer, nullable=False)
    endtime=db.Column(db.DateTime, default=datetime.utcnow().astimezone(timezone('Asia/Kolkata')))  #.astimezone(timezone('Asia/Kolkata')).strftime("%b %d %Y %H:%M:%S")

class exercise_type(db.Model):
    srno=db.Column(db.Integer,primary_key=True)
    exercise_id=db.Column(db.Integer,db.ForeignKey('exercise.exercise_id'))
    aasan_name=db.Column(db.String(200),nullable=False)
    aasan_duration=db.Column(db.Integer,nullable=False)
    

def generate_frames():
    fps=int(camera.get(cv2.CAP_PROP_FPS))
    print('fps=',fps)
    n=0
    
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        if(n%fps==0):
                captured_frames.append(frame)
        n+=1
        if not success:
            break
        else:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def predict():
    import os
    dict={}
    labels=['gomukhasana', 'natarajasana', 'padmasana', 'tadasana', 'vajrasana', 'vriksasana']

    # frames_path=r'C:\Users\UPLAV DANG\Desktop\python_webdev\Final_Project\Captured_frames'
    # for image in os.listdir(frames_path):
    #     img = os.path.join(frames_path+'/'+image)
    #     img = cv2.imread(img)
    #     img = cv2.resize(img,(256,256))
    #     test_img1 = np.asarray(img)
    #     test_img = test_img1.reshape(-1,256,256,3)
    #     captured.append(test_img)
    obj_ex=exercise(duration=len(captured_frames))
    db.session.add(obj_ex)
    db.session.commit()
    
    for img in captured_frames:
        img = cv2.resize(img,(256,256))
        test_img1 = np.asarray(img)
        test_img = test_img1.reshape(-1,256,256,3)
        p=new_model.predict(test_img)
        pose_name=labels[np.argmax(p)]
        probability=np.max(p)
        if(probability>0.4):
            if pose_name not in dict:
                dict[pose_name]=1
            else:
                dict[pose_name]+=1  
    for aasan in dict:
        obj_et=exercise_type(exercise_id=obj_ex.exercise_id, aasan_name=aasan, aasan_duration=dict[aasan])
        db.session.add(obj_et)    
    db.session.commit()


@app.route('/')
def func():
    # obj=exercise(duration=225)
    # db.session.add(obj)
    # db.session.commit()
    # obj2=exercise_type(exercise_id=obj.exercise_id, aasan_name='vrikshasaana', aasan_duration=115)
    # db.session.add(obj2)
    # db.session.commit()
    return render_template('home.html')

@app.route('/feed')
def feed():
    return render_template('feed.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/kill_feed')
def kill_feed():
    camera.release()
    predict()
    all_exercise=exercise.query.all()
    return render_template('summary.html',all_exercise=all_exercise)

@app.route('/list_exercise/<int:exercise_id>')
def show(exercise_id):
    data=exercise_type.query.filter_by(exercise_id=exercise_id)
    return render_template('list_of_exercise.html',data=data)
    

if __name__ == "__main__":
    app.run(debug=True)