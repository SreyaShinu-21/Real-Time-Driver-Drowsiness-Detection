from flask import Flask, render_template, Response, redirect, url_for, jsonify, request
import cv2, time, math, winsound
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import threading

app = Flask(__name__)

# Configuration - Best accuracy values
EAR_THRESHOLD = 0.18          # Best balance of sensitivity/accuracy
DROWSY_TIME = 3.0           # Exactly 3 seconds before first alert
CONTINUOUS_ALERT_INTERVAL = 2.0  # Alert every 2 seconds after first
BLINK_TIME = 0.3           # 0.3 seconds for normal blink
MIN_FACE_CONFIDENCE = 0.75   # Good balance for reliable detection
MODEL_PATH = "models/face_landmarker.task"

# Event logging
event_log = []

def log_event(message, event_type='info'):
    timestamp = time.strftime("%H:%M:%S", time.localtime())
    event_log.append({
        'time': timestamp,
        'message': message,
        'type': event_type
    })
    if len(event_log) > 100:
        event_log.pop(0)

def get_event_log():
    return event_log[-20:]

# Global variables
cap = None
face_landmarker = None
timestamp = 0
eye_closed_start = None
alarm_on = False
current_ear = 0.40
camera_lock = threading.Lock()

def is_detection_on():
    try:
        with open("detection_active.txt") as f:
            return f.read().strip() == "1"
    except:
        return False

# Simple camera initialization
def initialize_camera():
    global cap
    
    with camera_lock:
        if cap is not None:
            cap.release()
        
        print("🎥 Initializing camera...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = cap.read()
            if ret:
                avg_color = frame.mean()
                print(f"✅ Camera working! avg_color: {avg_color:.2f}")
                log_event("Camera initialized successfully", 'success')
                return True
            else:
                cap.release()
                cap = None
                print("❌ Camera opened but can't read frames")
        else:
            print("❌ Cannot open camera")
            cap = None
        
        log_event("Camera initialization failed", 'error')
        return False

# Face Landmarker initialization
def initialize_face_landmarker():
    global face_landmarker
    try:
        print("🤖 Initializing MediaPipe Face Landmarker...")
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=MIN_FACE_CONFIDENCE
        )
        face_landmarker = vision.FaceLandmarker.create_from_options(options)
        print("✅ Face Landmarker initialized successfully")
        log_event("Face Landmarker initialized successfully", 'success')
        return True
    except Exception as e:
        print(f"❌ Face Landmarker failed: {e}")
        log_event(f"Face Landmarker initialization failed: {str(e)}", 'error')
        face_landmarker = None
        return False

# Eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def calculate_ear(eye):
    A = math.dist(eye[1], eye[5])
    B = math.dist(eye[2], eye[4])
    C = math.dist(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Initialize everything
print("🚀 Starting SAFEDRIVE Driver Monitoring System...")
initialize_camera()
initialize_face_landmarker()

def generate_frames():
    global timestamp, eye_closed_start, alarm_on, current_ear, cap, face_landmarker
    
    print("📹 Starting frame generation...")
    
    while True:
        try:
            frame = None
            camera_working = False
            
            # Try to get camera frame
            with camera_lock:
                if cap is not None and cap.isOpened():
                    ret, test_frame = cap.read()
                    if ret:
                        avg_color = test_frame.mean()
                        if avg_color > 10:  # Good frame
                            frame = test_frame
                            camera_working = True
                        else:
                            print(f"❌ Frame too dark ({avg_color:.2f})")
                            cap.release()
                            cap = None
                    else:
                        print("❌ Failed to read frame")
                        cap.release()
                        cap = None
            
            # If camera failed, try to reinitialize
            if not camera_working:
                if initialize_camera():
                    continue  # Try again with new camera
                else:
                    # Create fallback frame
                    frame = create_fallback_frame()
            
            # Check detection status
            detection_active = is_detection_on()
            
            # Process face detection if active and camera is working
            if detection_active and face_landmarker is not None and camera_working:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = face_landmarker.detect_for_video(mp_image, timestamp)
                    timestamp += 1
                    
                    if result.face_landmarks:
                        h, w, _ = frame.shape
                        landmarks = result.face_landmarks[0]
                        
                        # Extract eye landmarks
                        left_eye, right_eye = [], []
                        for i in LEFT_EYE:
                            lm = landmarks[i]
                            left_eye.append((int(lm.x * w), int(lm.y * h)))
                        for i in RIGHT_EYE:
                            lm = landmarks[i]
                            right_eye.append((int(lm.x * w), int(lm.y * h)))
                        
                        # Calculate EAR
                        ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
                        current_ear = ear
                        
                        # Debug output
                        print(f"EAR: {ear:.3f}, Threshold: {EAR_THRESHOLD}")
                        print(f"Face detected: {len(result.face_landmarks) > 0}")
                        print(f"Face landmarks: {len(result.face_landmarks) if result.face_landmarks else 0}")
                        
                        # SIMPLE 3-SECOND DROWSINESS DETECTION
                        if ear < EAR_THRESHOLD:
                            print(f" Eyes closed! EAR: {ear:.3f} < {EAR_THRESHOLD}")
                            
                            # Start timing if not already started
                            if eye_closed_start is None:
                                eye_closed_start = time.time()
                                print(f"⏰ Started monitoring at {eye_closed_start}")
                                log_event("Eyes closed - monitoring for 3 seconds", 'warning')
                            
                            # Calculate how long eyes have been closed
                            eye_closed_duration = time.time() - eye_closed_start
                            print(f"⏱️ Eyes closed for: {eye_closed_duration:.1f}s")
                            
                            # Check if 3 seconds have passed
                            if eye_closed_duration >= DROWSY_TIME:
                                print(f"� 3 seconds passed! Triggering alerts...")
                                log_event("DROWSINESS ALERT - 3 seconds exceeded", 'error')
                                
                                # Show DROWSY status
                                cv2.putText(frame, "STATUS: DROWSY!", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.rectangle(frame, (10, 10), (200, 30), (0, 0, 255), -1)
                                cv2.putText(frame, "DETECTION: DROWSY", (15, 28), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                                cv2.rectangle(frame, (5, 5), (300, 100), (0, 0, 255), 3)
                                
                                # ALERT LOGIC: First alert + continuous
                                current_time = time.time()
                                
                                if 'last_alert_time' not in globals() or globals()['last_alert_time'] is None:
                                    # FIRST ALERT after exactly 3 seconds - Medium beep
                                    winsound.Beep(3000, 1000)  # Medium-high beep (3000Hz, 1s)
                                    globals()['last_alert_time'] = current_time
                                    print(f"FIRST ALERT at {eye_closed_duration:.1f}s")
                                    log_event("First drowsiness alert", 'error')
                                
                                elif current_time - globals()['last_alert_time'] >= CONTINUOUS_ALERT_INTERVAL:
                                    # CONTINUOUS ALERTS every 2 seconds - Long continuous beep
                                    if eye_closed_duration < 10:
                                        winsound.Beep(3500, 2000)  # Long continuous beep (3500Hz, 2s)
                                        print(f"CONTINUOUS ALERT at {eye_closed_duration:.1f}s")
                                        log_event("Continuous drowsiness alert", 'error')
                                    else:
                                        # EMERGENCY ALERT after 10 seconds - Double beep
                                        winsound.Beep(4000, 300)  # First emergency beep
                                        time.sleep(0.1)
                                        winsound.Beep(4000, 300)  # Second emergency beep
                                        print(f"EMERGENCY ALERT at {eye_closed_duration:.1f}s")
                                        log_event("Emergency drowsiness alert", 'critical')
                                    globals()['last_alert_time'] = current_time
                                
                            else:
                                # Eyes closed but not yet 3 seconds
                                print(f"⏳ Monitoring... {eye_closed_duration:.1f}s / 3.0s")
                                cv2.putText(frame, "STATUS: MONITORING", (20, 80), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                                cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                cv2.rectangle(frame, (10, 10), (200, 30), (255, 165, 0), -1)
                                cv2.putText(frame, "DETECTION: MONITOR", (15, 28), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        else:
                            # EYES ARE OPEN - Reset everything
                            print(f"👀 Eyes open! EAR: {ear:.3f}")
                            
                            # Reset all timers
                            eye_closed_start = None
                            if 'last_alert_time' in globals():
                                globals()['last_alert_time'] = None
                            
                            # Show AWAKE status
                            cv2.putText(frame, "STATUS: AWAKE", (20, 80), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.putText(frame, f"EAR: {ear:.2f}", (20, 40), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            cv2.rectangle(frame, (10, 10), (200, 30), (0, 255, 0), -1)
                            cv2.putText(frame, "DETECTION: ACTIVE", (15, 28), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                        
                        # Draw face landmarks
                        for i in LEFT_EYE + RIGHT_EYE:
                            lm = landmarks[i]
                            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)
                    else:
                        cv2.putText(frame, "NO FACE DETECTED", (20, 40), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        log_event("No face detected in frame", 'warning')
                        
                except Exception as e:
                    print(f"❌ Face detection error: {e}")
                    log_event(f"Face detection error: {str(e)}", 'error')
                    
            else:
                # Show idle status
                cv2.rectangle(frame, (10, 10), (200, 30), (255, 165, 0), -1)
                cv2.putText(frame, "DETECTION: IDLE", (15, 28), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, f"EAR: {current_ear:.2f}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                if camera_working:
                    cv2.putText(frame, "STATUS: CAMERA READY", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "STATUS: CAMERA INITIALIZING", (20, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            
            # Add timestamp
            current_time = time.strftime("%H:%M:%S", time.localtime())
            cv2.putText(frame, current_time, (frame.shape[1]-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add camera status
            if camera_working:
                cv2.putText(frame, "CAMERA: CONNECTED", (frame.shape[1]-200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "CAMERA: RECONNECTING...", (frame.shape[1]-200, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
            
            # Encode and send frame
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            else:
                print("❌ Failed to encode frame")
                
        except Exception as e:
            print(f"❌ Frame generation error: {e}")
            log_event(f"Frame generation error: {str(e)}", 'error')
            time.sleep(0.1)
            continue

def create_fallback_frame():
    """Create animated fallback frame when camera is not available"""
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Create animated gradient
    t = time.time()
    for i in range(480):
        color_val = int(128 + 127 * math.sin(i/50 + t))
        frame[i, :] = [color_val//2, 100, 255-color_val//2]
    
    # Add text
    cv2.putText(frame, "SAFEDRIVE MONITORING", (80, 150), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(frame, "Camera Initializing...", (140, 200), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "System is Working", (170, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Add current EAR
    cv2.putText(frame, f"EAR: {current_ear:.2f}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "STATUS: INITIALIZING", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
    
    return frame

def get_current_ear():
    return current_ear

def test_camera():
    """Test camera functionality"""
    with camera_lock:
        if cap is None:
            return {"status": "error", "message": "Camera not initialized"}
        
        if not cap.isOpened():
            return {"status": "error", "message": "Camera not accessible"}
        
        try:
            ret, frame = cap.read()
            if not ret:
                return {"status": "error", "message": "Cannot read camera frames"}
            
            avg_color = frame.mean()
            return {
                "status": "success", 
                "message": "Camera working properly", 
                "resolution": frame.shape[:2], 
                "avg_color": float(avg_color)
            }
        except Exception as e:
            return {"status": "error", "message": f"Camera test failed: {str(e)}"}

# Flask Routes
@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/start')
def start():
    open("detection_active.txt", "w").write("1")
    log_event("Detection system started by user", 'success')
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    open("detection_active.txt", "w").write("0")
    log_event("Detection system halted by user", 'info')
    return redirect(url_for('index'))

@app.route('/status')
def status():
    with camera_lock:
        camera_status = cap is not None and cap.isOpened() if 'cap' in globals() else False
    return jsonify({
        'detection_active': is_detection_on(),
        'ear': get_current_ear(),
        'camera_connected': camera_status
    })

@app.route('/test_camera')
def test_camera_endpoint():
    return jsonify(test_camera())

@app.route('/events')
def events():
    return jsonify({
        'events': get_event_log()
    })

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
