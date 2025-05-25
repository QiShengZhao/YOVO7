import os
import cv2
import torch
import numpy as np
import json
import time
import uuid
import base64
import requests
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response, send_file, render_template, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import threading
import sqlite3
from functools import wraps
import secrets
import smtplib
from email.message import EmailMessage
from flask_sqlalchemy import SQLAlchemy
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature

app = Flask(__name__, static_folder='.', static_url_path='')

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
RESULT_FOLDER = 'results'
DATABASE_FOLDER = 'database'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MODEL_EXTENSIONS = {'pt', 'pth', 'onnx', 'torchscript'}

# DeepSeek API配置
DEEPSEEK_API_KEY = 'sk-2add3be9c4d44cd98817e0161f65b601'
DEEPSEEK_API_URL = 'https://api.deepseek.com/v1/chat/completions'
DEEPSEEK_MODEL = 'deepseek-chat'  # 使用deepseek-chat模型，支持视觉能力

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# App configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(16))
# Use absolute path for SQLite database
basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, DATABASE_FOLDER, 'app.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['MAIL_SERVER'] = os.environ.get('MAIL_SERVER', 'smtp.example.com')
app.config['MAIL_PORT'] = int(os.environ.get('MAIL_PORT', 587))
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', 'user@example.com')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', 'password')
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_DEFAULT_SENDER'] = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@example.com')
app.config['SECURITY_PASSWORD_SALT'] = os.environ.get('SECURITY_PASSWORD_SALT', secrets.token_hex(16))

# Initialize SQLAlchemy
db = SQLAlchemy(app)

# Email token serializer
ts = URLSafeTimedSerializer(app.config["SECRET_KEY"])

# User database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    detections = db.relationship('Detection', backref='user', lazy=True)
    ai_analyses = db.relationship('AIAnalysis', backref='user', lazy=True)

    def __repr__(self):
        return f'<User {self.username}>'

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Detection(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    image_path = db.Column(db.String(255), nullable=True)
    result_path = db.Column(db.String(255), nullable=True)
    num_defects = db.Column(db.Integer, default=0)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    details = db.Column(db.Text, nullable=True)  # JSON string of detection details

# 新增: DeepSeek AI分析结果模型
class AIAnalysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    detection_id = db.Column(db.Integer, db.ForeignKey('detection.id'), nullable=True)
    image_path = db.Column(db.String(255), nullable=True)
    analysis_text = db.Column(db.Text, nullable=False)  # AI分析结果
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # 与Detection模型的关系
    detection = db.relationship('Detection', backref='ai_analyses')

# Global variables
current_model = None
video_capture = None
video_thread = None
processing_frame = False
stop_video = False
stats = {
    'total_detections': 0,
    'total_defects': 0
}
detection_history = []

# Helper functions
def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def get_file_size(file_path):
    size_bytes = os.path.getsize(file_path)
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"

def get_model_type(file_path):
    extension = file_path.rsplit('.', 1)[1].lower()
    if extension == 'pt' or extension == 'pth':
        return 'PyTorch'
    elif extension == 'onnx':
        return 'ONNX'
    elif extension == 'torchscript':
        return 'TorchScript'
    return 'Unknown'

# Email functions
def send_email(to, subject, body):
    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg['Subject'] = subject
        msg['From'] = app.config['MAIL_DEFAULT_SENDER']
        msg['To'] = to
        
        # Connect to server and send email
        server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
        if app.config['MAIL_USE_TLS']:
            server.starttls()
        server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        print(f"Failed to send email: {e}")
        return False

def send_confirmation_email(user_email):
    token = ts.dumps(user_email, salt=app.config['SECURITY_PASSWORD_SALT'])
    confirm_url = url_for('confirm_email', token=token, _external=True)
    subject = "请确认您的邮箱"
    body = f"""
    感谢您注册工业产品缺陷检测系统！
    请点击以下链接确认您的邮箱地址：
    
    {confirm_url}
    
    此链接有效期为24小时。
    
    如果您没有注册本系统，请忽略此邮件。
    """
    return send_email(user_email, subject, body)

# Authentication decorators
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'error', 'message': '请先登录'}), 401
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'status': 'error', 'message': '请先登录'}), 401
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            return jsonify({'status': 'error', 'message': '需要管理员权限'}), 403
        
        return f(*args, **kwargs)
    return decorated_function

# Mock detection function (replace with actual model inference)
def detect_defects(image, confidence_threshold=0.5):
    # This is a mock function - replace with actual model inference
    height, width = image.shape[:2]
    
    # Simulate processing time
    time.sleep(0.2)
    
    # Generate random detections for demonstration
    num_detections = np.random.randint(0, 5)
    detections = []
    
    defect_classes = ['划痕', '凹陷', '裂缝', '污渍', '变形']
    
    for _ in range(num_detections):
        class_id = np.random.randint(0, len(defect_classes))
        confidence = np.random.uniform(confidence_threshold, 1.0)
        
        # Random bounding box
        x1 = np.random.randint(0, width - 100)
        y1 = np.random.randint(0, height - 100)
        w = np.random.randint(50, 200)
        h = np.random.randint(50, 200)
        x2 = min(x1 + w, width)
        y2 = min(y1 + h, height)
        
        detections.append({
            'class_id': class_id,
            'class_name': defect_classes[class_id],
            'confidence': float(confidence),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        })
        
        # Draw bounding box on the image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{defect_classes[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return image, detections

# Video processing thread
def process_video():
    global video_capture, processing_frame, stop_video, stats
    
    while video_capture is not None and video_capture.isOpened() and not stop_video:
        if not processing_frame:
            processing_frame = True
            ret, frame = video_capture.read()
            
            if not ret:
                processing_frame = False
                break
            
            # Process the frame
            processed_frame, detections = detect_defects(frame)
            
            # Update statistics
            stats['total_detections'] += 1
            stats['total_defects'] += len(detections)
            
            # Add to history
            if len(detections) > 0:
                detection_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'detections': detections
                }
                detection_history.append(detection_entry)
                
                # Save to database if user is logged in
                if 'user_id' in session:
                    try:
                        detection_db = Detection(
                            user_id=session['user_id'],
                            num_defects=len(detections),
                            details=json.dumps(detections)
                        )
                        db.session.add(detection_db)
                        db.session.commit()
                    except Exception as e:
                        print(f"Failed to save detection: {e}")
                        db.session.rollback()
            
            # Encode the frame
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            
            # Stream the frame
            app.config['current_frame'] = frame
            processing_frame = False
    
    # Clean up
    if video_capture is not None:
        video_capture.release()
    video_capture = None
    app.config['current_frame'] = None

# Routes
@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return app.send_static_file('industrial-dashboard-html.html')

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return app.send_static_file('login.html')
    
    username = request.json.get('username')
    password = request.json.get('password')
    
    if not username or not password:
        return jsonify({'status': 'error', 'message': '用户名和密码是必填的'}), 400
    
    user = User.query.filter_by(username=username).first()
    
    if not user or not user.check_password(password):
        return jsonify({'status': 'error', 'message': '用户名或密码错误'}), 401
    
    if not user.is_active:
        return jsonify({'status': 'error', 'message': '账号未激活，请检查邮箱进行确认'}), 403
    
    session['user_id'] = user.id
    session['is_admin'] = user.is_admin
    
    return jsonify({
        'status': 'success', 
        'message': '登录成功',
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin
        }
    })

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return app.send_static_file('register.html')
    
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not username or not email or not password:
            return jsonify({'status': 'error', 'message': '所有字段都是必填的'}), 400
        
        # Check if user exists
        existing_user = User.query.filter((User.username == username) | (User.email == email)).first()
        if existing_user:
            return jsonify({'status': 'error', 'message': '用户名或邮箱已存在'}), 400
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        
        # Check if this is the first user (make them admin)
        if User.query.count() == 0:
            user.is_admin = True
            user.is_active = True  # First user is auto-activated
        
        db.session.add(user)
        
        try:
            db.session.commit()
            # Send confirmation email
            if not user.is_active:
                send_confirmation_email(user.email)
            return jsonify({'status': 'success', 'message': '注册成功，请检查邮箱进行确认'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'status': 'error', 'message': f'注册失败: {str(e)}'}), 500
    
    return jsonify({'status': 'error', 'message': '仅支持GET和POST请求'}), 405

@app.route('/confirm/<token>')
def confirm_email(token):
    try:
        email = ts.loads(token, salt=app.config['SECURITY_PASSWORD_SALT'], max_age=86400)  # 24 hours
    except SignatureExpired:
        return jsonify({'status': 'error', 'message': '确认链接已过期'}), 400
    except BadSignature:
        return jsonify({'status': 'error', 'message': '确认链接无效'}), 400
    
    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 400
    
    user.is_active = True
    db.session.commit()
    
    return jsonify({'status': 'success', 'message': '邮箱确认成功，现在可以登录了'})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user_id', None)
    session.pop('is_admin', None)
    return jsonify({'status': 'success', 'message': '已退出登录'})

@app.route('/profile', methods=['GET'])
@login_required
def get_profile():
    if request.headers.get('Content-Type') == 'application/json':
        user = User.query.get(session['user_id'])
        if not user:
            return jsonify({'status': 'error', 'message': '用户不存在'}), 404
        
        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_admin': user.is_admin,
                'created_at': user.created_at.isoformat()
            }
        })
    else:
        # Serve the HTML page for browser requests
        return app.send_static_file('profile.html')

@app.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    
    data = request.json
    
    if 'username' in data and data['username'] != user.username:
        existing_user = User.query.filter_by(username=data['username']).first()
        if existing_user:
            return jsonify({'status': 'error', 'message': '用户名已存在'}), 400
        user.username = data['username']
    
    if 'email' in data and data['email'] != user.email:
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user:
            return jsonify({'status': 'error', 'message': '邮箱已存在'}), 400
        user.email = data['email']
        user.is_active = False  # Require re-confirmation
        send_confirmation_email(user.email)
    
    if 'password' in data:
        user.set_password(data['password'])
    
    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': '个人资料已更新'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'更新失败: {str(e)}'}), 500

# Admin user management routes
@app.route('/api/users', methods=['GET'])
@admin_required
def get_users():
    users = User.query.all()
    return jsonify({
        'status': 'success',
        'users': [{
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat()
        } for user in users]
    })

@app.route('/api/users/<int:user_id>', methods=['GET'])
@admin_required
def get_user(user_id):
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    
    return jsonify({
        'status': 'success',
        'user': {
            'id': user.id,
            'username': user.username,
            'email': user.email,
            'is_admin': user.is_admin,
            'is_active': user.is_active,
            'created_at': user.created_at.isoformat(),
            'detections_count': Detection.query.filter_by(user_id=user.id).count()
        }
    })

@app.route('/api/users/<int:user_id>', methods=['PUT'])
@admin_required
def update_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'status': 'error', 'message': '不能修改自己的管理员状态'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    
    data = request.json
    
    if 'is_admin' in data:
        user.is_admin = bool(data['is_admin'])
    
    if 'is_active' in data:
        user.is_active = bool(data['is_active'])
    
    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': '用户已更新'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'更新失败: {str(e)}'}), 500

@app.route('/api/users/<int:user_id>', methods=['DELETE'])
@admin_required
def delete_user(user_id):
    if user_id == session['user_id']:
        return jsonify({'status': 'error', 'message': '不能删除自己的账号'}), 400
    
    user = User.query.get(user_id)
    if not user:
        return jsonify({'status': 'error', 'message': '用户不存在'}), 404
    
    try:
        db.session.delete(user)
        db.session.commit()
        return jsonify({'status': 'success', 'message': '用户已删除'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': f'删除失败: {str(e)}'}), 500

# API routes
@app.route('/api/models', methods=['GET'])
@login_required
def get_models():
    models = []
    
    for filename in os.listdir(MODEL_FOLDER):
        if allowed_file(filename, MODEL_EXTENSIONS):
            file_path = os.path.join(MODEL_FOLDER, filename)
            models.append({
                'name': filename,
                'path': file_path,
                'type': get_model_type(file_path),
                'size': get_file_size(file_path)
            })
    
    return jsonify({
        'status': 'success',
        'models': models,
        'current_model': current_model
    })

@app.route('/api/load_model', methods=['POST'])
@login_required
def load_model():
    global current_model
    
    # Check if user is admin for model loading
    user = User.query.get(session['user_id'])
    if not user.is_admin:
        return jsonify({
            'status': 'error',
            'message': '只有管理员可以加载模型'
        }), 403
    
    data = request.json
    model_path = data.get('model_path')
    
    if not model_path or not os.path.exists(model_path):
        return jsonify({
            'status': 'error',
            'message': '无效的模型路径'
        })
    
    try:
        # In a real implementation, you would load the model here
        # model = torch.load(model_path)
        
        model_info = {
            'name': os.path.basename(model_path),
            'path': model_path,
            'type': get_model_type(model_path),
            'size': get_file_size(model_path),
            'loaded_at': datetime.now().isoformat(),
            'loaded_by': user.username
        }
        
        current_model = model_info
        
        return jsonify({
            'status': 'success',
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/upload_model', methods=['POST'])
@login_required
def upload_model():
    # Check if user is admin for model upload
    user = User.query.get(session['user_id'])
    if not user.is_admin:
        return jsonify({
            'status': 'error',
            'message': '只有管理员可以上传模型'
        }), 403
    
    if 'model' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '没有提供模型文件'
        })
    
    file = request.files['model']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': '未选择文件'
        })
    
    if file and allowed_file(file.filename, MODEL_EXTENSIONS):
        filename = secure_filename(file.filename)
        file_path = os.path.join(MODEL_FOLDER, filename)
        file.save(file_path)
        
        return jsonify({
            'status': 'success',
            'file_path': file_path,
            'message': '模型上传成功'
        })
    
    return jsonify({
        'status': 'error',
        'message': '无效的文件类型'
    })

@app.route('/api/start_video', methods=['POST'])
@login_required
def start_video():
    global video_capture, video_thread, stop_video
    
    if video_capture is not None:
        return jsonify({
            'status': 'error',
            'message': '视频已在运行中'
        })
    
    data = request.json
    source = data.get('source', '0')
    
    try:
        # Try to convert to integer for webcam
        source = int(source)
    except ValueError:
        # Keep as string for video file path
        pass
    
    try:
        video_capture = cv2.VideoCapture(source)
        
        if not video_capture.isOpened():
            return jsonify({
                'status': 'error',
                'message': '无法打开视频源'
            })
        
        # Set initial frame
        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            video_capture = None
            return jsonify({
                'status': 'error',
                'message': '无法从视频源读取'
            })
        
        app.config['current_frame'] = None
        stop_video = False
        
        # Start video processing thread
        video_thread = threading.Thread(target=process_video)
        video_thread.daemon = True
        video_thread.start()
        
        return jsonify({
            'status': 'success',
            'message': '视频成功启动'
        })
    except Exception as e:
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/api/stop_video', methods=['POST'])
@login_required
def stop_video():
    global video_capture, stop_video
    
    if video_capture is None:
        return jsonify({
            'status': 'error',
            'message': '没有正在运行的视频'
        })
    
    stop_video = True
    
    return jsonify({
        'status': 'success',
        'message': '视频已成功停止'
    })

@app.route('/video_feed')
@login_required
def video_feed():
    def generate():
        while True:
            if 'current_frame' in app.config and app.config['current_frame'] is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + app.config['current_frame'] + b'\r\n')
            else:
                time.sleep(0.1)
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detect', methods=['POST'])
@login_required
def detect():
    if 'image' not in request.files:
        return jsonify({
            'status': 'error',
            'message': '没有提供图片文件'
        })
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'status': 'error',
            'message': '未选择文件'
        })
    
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        # Save the uploaded image
        filename = f"{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        # Read the image
        image = cv2.imread(upload_path)
        
        # Get confidence threshold
        confidence = float(request.form.get('confidence', 0.5))
        
        # Process the image
        processed_image, detections = detect_defects(image, confidence)
        
        # Save the result
        result_filename = f"result_{filename}"
        result_path = os.path.join(RESULT_FOLDER, result_filename)
        cv2.imwrite(result_path, processed_image)
        
        # Update statistics
        stats['total_detections'] += 1
        stats['total_defects'] += len(detections)
        
        # Save to database
        user_id = session['user_id']
        detection_db = Detection(
            user_id=user_id,
            image_path=upload_path,
            result_path=result_path,
            num_defects=len(detections),
            details=json.dumps(detections)
        )
        db.session.add(detection_db)
        db.session.commit()
        
        # Add to history
        if len(detections) > 0:
            detection_entry = {
                'timestamp': datetime.now().isoformat(),
                'image': filename,
                'detections': detections,
                'user_id': user_id
            }
            detection_history.append(detection_entry)
        
        return jsonify({
            'status': 'success',
            'result_image': f"/results/{result_filename}",
            'detections': detections,
            'total_defects': len(detections)
        })
    
    return jsonify({
        'status': 'error',
        'message': '无效的文件类型'
    })

@app.route('/api/stats', methods=['GET'])
@login_required
def get_stats():
    # For admin, return global stats
    if session.get('is_admin'):
        return jsonify({
            'status': 'success',
            'stats': stats
        })
    
    # For regular users, return their own stats
    user_id = session['user_id']
    user_detections = Detection.query.filter_by(user_id=user_id).all()
    user_stats = {
        'total_detections': len(user_detections),
        'total_defects': sum(d.num_defects for d in user_detections)
    }
    
    return jsonify({
        'status': 'success',
        'stats': user_stats
    })

@app.route('/api/user_detections', methods=['GET'])
@login_required
def get_user_detections():
    user_id = session['user_id']
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    query = Detection.query.filter_by(user_id=user_id).order_by(Detection.timestamp.desc())
    pagination = query.paginate(page=page, per_page=per_page)
    
    detections = [{
        'id': d.id,
        'timestamp': d.timestamp.isoformat(),
        'image_path': d.image_path,
        'result_path': d.result_path,
        'num_defects': d.num_defects,
        'details': json.loads(d.details) if d.details else None
    } for d in pagination.items]
    
    return jsonify({
        'status': 'success',
        'detections': detections,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'total': pagination.total,
        'pages': pagination.pages
    })

@app.route('/api/export_data', methods=['GET'])
@login_required
def export_data():
    user = User.query.get(session['user_id'])
    
    # Determine which data to export
    if user.is_admin and request.args.get('all') == 'true':
        # Admin can export all data
        detections_to_export = Detection.query.all()
        filename_prefix = 'all'
    else:
        # Regular users export only their data
        detections_to_export = Detection.query.filter_by(user_id=user.id).all()
        filename_prefix = f'user_{user.id}'
    
    # Format data
    export_data = {
        'stats': {
            'total_detections': len(detections_to_export),
            'total_defects': sum(d.num_defects for d in detections_to_export)
        },
        'history': [{
            'id': d.id,
            'timestamp': d.timestamp.isoformat(),
            'user': user.username,
            'num_defects': d.num_defects,
            'details': json.loads(d.details) if d.details else None
        } for d in detections_to_export],
        'exported_at': datetime.now().isoformat(),
        'exported_by': user.username
    }
    
    # Create export file
    export_filename = f"detection_export_{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    export_path = os.path.join(RESULT_FOLDER, export_filename)
    
    with open(export_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    return send_file(export_path, as_attachment=True)

# DeepSeek AI分析函数
def analyze_image_with_deepseek(image_path, defects_info=None):
    """使用DeepSeek AI分析图像中的缺陷
    
    注意: 当前DeepSeek API不支持直接通过base64传输图像，
    此函数将只处理检测到的缺陷信息，不发送图像。
    """
    try:
        # 准备提示信息
        if defects_info:
            prompt = f"""
            请分析工业产品中检测到的缺陷。以下是检测到的缺陷信息:
            {json.dumps(defects_info, ensure_ascii=False, indent=2)}
            
            请提供以下分析:
            1. 缺陷的严重程度评估
            2. 可能的缺陷原因
            3. 建议的修复方法
            4. 预防此类缺陷的生产建议
            
            请用中文回答，并保持专业的工程分析风格。
            """
        else:
            prompt = """
            无法获取缺陷信息。请提供一个通用的工业产品缺陷分析，包括:
            
            1. 常见的工业产品缺陷类型
            2. 缺陷产生的一般原因
            3. 预防和检测缺陷的方法
            4. 质量控制建议
            
            请用中文回答，并保持专业的工程分析风格。
            """
        
        # 构建请求负载
        headers = {
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": DEEPSEEK_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        # 发送请求到DeepSeek API
        print(f"发送请求到DeepSeek API...")
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        
        # 检查HTTP状态码
        if response.status_code != 200:
            error_message = f"DeepSeek API返回错误状态码: {response.status_code}"
            print(error_message)
            if response.text:
                print(f"错误详情: {response.text}")
            return f"AI分析失败: API返回错误 ({response.status_code})"
        
        # 尝试解析JSON响应
        try:
            response_data = response.json()
            print(f"成功获取DeepSeek API响应")
        except json.JSONDecodeError as e:
            error_message = f"无法解析API响应JSON: {str(e)}, 响应内容: {response.text[:200]}..."
            print(error_message)
            return f"AI分析失败: 无法解析API响应 ({str(e)})"
        
        # 检查API响应格式
        if 'choices' in response_data and len(response_data['choices']) > 0:
            analysis_text = response_data['choices'][0]['message']['content']
            return analysis_text
        else:
            error_message = f"API响应格式无效: {json.dumps(response_data)[:200]}..."
            print(error_message)
            return "AI分析失败，API返回格式不符合预期。"
            
    except requests.exceptions.Timeout:
        error_message = "DeepSeek API请求超时"
        print(error_message)
        return "AI分析失败: API请求超时，请稍后重试。"
    except requests.exceptions.ConnectionError:
        error_message = "连接DeepSeek API失败"
        print(error_message)
        return "AI分析失败: 无法连接到API服务器，请检查网络连接。"
    except Exception as e:
        error_message = f"DeepSeek分析出错: {str(e)}"
        print(error_message)
        import traceback
        traceback.print_exc()
        return f"AI分析过程中出现错误: {str(e)}"

# 新增API端点: AI分析已检测的图像
@app.route('/api/analyze_detection/<int:detection_id>', methods=['POST'])
@login_required
def analyze_detection(detection_id):
    # 获取检测记录
    detection = Detection.query.get(detection_id)
    if not detection:
        return jsonify({'status': 'error', 'message': '未找到指定的检测记录'}), 404
    
    # 检查权限 (只能分析自己的检测或管理员权限)
    if detection.user_id != session['user_id'] and not session.get('is_admin'):
        return jsonify({'status': 'error', 'message': '没有权限分析此检测记录'}), 403
    
    # 获取检测详情
    defects_info = None
    if detection.details:
        try:
            defects_info = json.loads(detection.details)
        except:
            pass
    
    # 使用DeepSeek分析图像
    if detection.image_path and os.path.exists(detection.image_path):
        analysis_result = analyze_image_with_deepseek(detection.image_path, defects_info)
        
        # 保存分析结果到数据库
        ai_analysis = AIAnalysis(
            detection_id=detection.id,
            image_path=detection.image_path,
            analysis_text=analysis_result,
            user_id=session['user_id']
        )
        db.session.add(ai_analysis)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'analysis_id': ai_analysis.id,
            'analysis_text': analysis_result,
            'detection_id': detection_id
        })
    else:
        return jsonify({'status': 'error', 'message': '检测记录中没有可用的图像'}), 400

# 新增API端点: 获取分析历史
@app.route('/api/analysis_history', methods=['GET'])
@login_required
def get_analysis_history():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # 根据用户权限查询
    if session.get('is_admin') and request.args.get('all') == 'true':
        query = AIAnalysis.query.order_by(AIAnalysis.timestamp.desc())
    else:
        query = AIAnalysis.query.filter_by(user_id=session['user_id']).order_by(AIAnalysis.timestamp.desc())
    
    # 分页
    pagination = query.paginate(page=page, per_page=per_page)
    
    analyses = [{
        'id': analysis.id,
        'timestamp': analysis.timestamp.isoformat(),
        'detection_id': analysis.detection_id,
        'image_path': analysis.image_path,
        'analysis_text': analysis.analysis_text
    } for analysis in pagination.items]
    
    return jsonify({
        'status': 'success',
        'analyses': analyses,
        'page': pagination.page,
        'per_page': pagination.per_page,
        'total': pagination.total,
        'pages': pagination.pages
    })

# 新增API端点: 直接分析上传的图像
@app.route('/api/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': '没有提供图片文件'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': '未选择文件'}), 400
    
    if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
        # 保存上传的图像
        filename = f"{uuid.uuid4().hex}.jpg"
        upload_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(upload_path)
        
        # 使用DeepSeek分析图像
        analysis_result = analyze_image_with_deepseek(upload_path)
        
        # 保存分析结果到数据库
        ai_analysis = AIAnalysis(
            image_path=upload_path,
            analysis_text=analysis_result,
            user_id=session['user_id']
        )
        db.session.add(ai_analysis)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'analysis_id': ai_analysis.id,
            'analysis_text': analysis_result,
            'image_path': upload_path
        })
    
    return jsonify({'status': 'error', 'message': '无效的文件类型'}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        # Create admin user if no users exist
        if User.query.count() == 0:
            admin = User(
                username="admin",
                email="admin@example.com",
                is_admin=True,
                is_active=True
            )
            admin.set_password("admin123")  # Default password, should be changed
            db.session.add(admin)
            db.session.commit()
            print("Created default admin user")
    app.run(debug=True, host='0.0.0.0', port=5000) 