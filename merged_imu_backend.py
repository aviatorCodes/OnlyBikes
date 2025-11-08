import requests
import time
import os
import csv
import numpy as np
from collections import deque
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple, List
from flask import Flask, jsonify
from flask_cors import CORS
import threading

# ----------------------------------------------------------------- 
# PHONE URL CONFIGURATION
# ----------------------------------------------------------------- 
PHONE_URL = "http://172.16.88.240:8080"
SENSORS_TO_GET = ["accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ", "lat", "lon", "altitude"]
FULL_REQUEST_URL = f"{PHONE_URL}/get?{'&'.join(SENSORS_TO_GET)}"

# ----------------------------------------------------------------- 
@dataclass
class IMUState:
    """State vector for IMU navigation"""
    position: np.ndarray
    velocity: np.ndarray
    orientation: Rotation
    acc_bias: np.ndarray
    gyr_bias: np.ndarray

class AdvancedDeadReckoning:
    def __init__(self, dt: float = 0.5, initial_heading: float = 0.0):
        self.dt = dt
        self.g = 9.81
        
        self.state = IMUState(
            position=np.zeros(3),
            velocity=np.zeros(3),
            orientation=Rotation.from_euler('z', initial_heading),
            acc_bias=np.zeros(3),
            gyr_bias=np.zeros(3)
        )
        
        self.P = np.eye(15) * 0.1
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 
                         0.001, 0.001, 0.001, 0.0001, 0.0001, 0.0001])
        
        self.zupt_threshold_acc = 0.5
        self.zupt_threshold_gyr = 0.1
        self.zupt_window = 5
        self.step_threshold = 0.8
        self.min_step_interval = 10
        
        self.history = {
            'position': [],
            'velocity': [],
            'acc_magnitude': [],
            'zupt_detected': [],
            'steps_detected': []
        }
    
    def detect_zero_velocity(self, acc: np.ndarray, gyr: np.ndarray, 
                            window_data: List[Tuple[np.ndarray, np.ndarray]]) -> bool:
        if len(window_data) < self.zupt_window:
            return False
        
        acc_mag = np.linalg.norm(acc)
        gyr_mag = np.linalg.norm(gyr)
        
        acc_window = np.array([np.linalg.norm(a) for a, _ in window_data[-self.zupt_window:]])
        gyr_window = np.array([np.linalg.norm(g) for _, g in window_data[-self.zupt_window:]])
        
        acc_var = np.var(acc_window)
        gyr_var = np.var(gyr_window)
        
        return (acc_var < 0.05 and gyr_var < 0.001 and 
                abs(acc_mag - self.g) < self.zupt_threshold_acc and 
                gyr_mag < self.zupt_threshold_gyr)
    
    def apply_zupt(self):
        H = np.zeros((3, 15))
        H[0:3, 3:6] = np.eye(3)
        R = np.eye(3) * 0.001
        
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        innovation = -self.state.velocity
        
        delta_x = K @ innovation
        self.state.velocity += delta_x[3:6]
        self.state.velocity *= 0.1
        self.P = (np.eye(15) - K @ H) @ self.P
    
    def detect_steps(self, acc_vertical_history: List[float]) -> bool:
        if len(acc_vertical_history) < 20:
            return False
        recent = np.array(acc_vertical_history[-20:])
        peaks, _ = find_peaks(recent, height=self.step_threshold, distance=self.min_step_interval)
        return len(peaks) > 0 and peaks[-1] >= len(recent) - 3
    
    def compensate_gravity(self, acc_body: np.ndarray) -> np.ndarray:
        gravity_global = np.array([0, 0, -self.g])
        gravity_body = self.state.orientation.inv().apply(gravity_global)
        return acc_body - gravity_body
    
    def update_orientation(self, gyr: np.ndarray):
        gyr_corrected = gyr - self.state.gyr_bias
        angle = np.linalg.norm(gyr_corrected) * self.dt
        if angle > 1e-8:
            axis = gyr_corrected / np.linalg.norm(gyr_corrected)
            delta_rot = Rotation.from_rotvec(axis * angle)
            self.state.orientation = self.state.orientation * delta_rot
    
    def update_velocity_position(self, acc_body: np.ndarray):
        acc_corrected = self.compensate_gravity(acc_body - self.state.acc_bias)
        acc_global = self.state.orientation.apply(acc_corrected)
        self.state.velocity += acc_global * self.dt
        self.state.position += self.state.velocity * self.dt
    
    def adaptive_bias_estimation(self, acc: np.ndarray, gyr: np.ndarray, is_stationary: bool):
        if is_stationary:
            gravity_body = self.state.orientation.inv().apply(np.array([0, 0, -self.g]))
            acc_error = acc - gravity_body
            self.state.acc_bias += 0.01 * acc_error
            self.state.gyr_bias += 0.01 * gyr
    
    def process_measurement(self, acc: np.ndarray, gyr: np.ndarray, 
                          window_data: List, acc_vert_history: List) -> dict:
        is_stationary = self.detect_zero_velocity(acc, gyr, window_data)
        step_detected = self.detect_steps(acc_vert_history)
        
        self.update_orientation(gyr)
        self.adaptive_bias_estimation(acc, gyr, is_stationary)
        
        if is_stationary:
            self.apply_zupt()
        
        self.update_velocity_position(acc)
        
        acc_magnitude = np.linalg.norm(acc)
        self.history['position'].append(self.state.position.copy())
        self.history['velocity'].append(np.linalg.norm(self.state.velocity))
        self.history['acc_magnitude'].append(acc_magnitude)
        self.history['zupt_detected'].append(is_stationary)
        self.history['steps_detected'].append(step_detected)
        
        return {
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'is_stationary': is_stationary,
            'step_detected': step_detected
        }

# ----------------------------------------------------------------- 
# MOTION CLASSIFIER
# ----------------------------------------------------------------- 
class MotionClassifier:
    def __init__(self, buffer_size: int = 10):
        self.buffer_size = buffer_size
        self.data_buffer = {sensor: deque(maxlen=buffer_size) for sensor in SENSORS_TO_GET}
    
    def add_data(self, data):
        for sensor in SENSORS_TO_GET:
            if data.get(sensor) is not None:
                self.data_buffer[sensor].append(data[sensor])
    
    def analyze_motion(self):
        if len(self.data_buffer['accX']) < 5:
            return "COLLECTING DATA...", {}
        
        acc_x = np.array(list(self.data_buffer['accX']))
        acc_y = np.array(list(self.data_buffer['accY']))
        acc_z = np.array(list(self.data_buffer['accZ']))
        gyr_x = np.array(list(self.data_buffer['gyrX']))
        gyr_y = np.array(list(self.data_buffer['gyrY']))
        gyr_z = np.array(list(self.data_buffer['gyrZ']))
        
        acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
        gyro_magnitude = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
        
        acc_variance = np.var(acc_magnitude)
        acc_std = np.std(acc_magnitude)
        acc_mean = np.mean(acc_magnitude)
        gyro_variance = np.var(gyro_magnitude)
        gyro_mean = np.mean(gyro_magnitude)
        
        acc_diff = np.diff(acc_magnitude)
        noise_level = np.std(acc_diff) if len(acc_diff) > 0 else 0
        
        position_change = 0
        if self.data_buffer['lat'] and len(self.data_buffer['lat']) >= 2:
            lat_change = abs(list(self.data_buffer['lat'])[-1] - list(self.data_buffer['lat'])[0])
            lon_change = abs(list(self.data_buffer['lon'])[-1] - list(self.data_buffer['lon'])[0])
            position_change = np.sqrt(lat_change**2 + lon_change**2)
        
        activity = self.classify_activity(acc_variance, acc_mean, acc_std, 
                                         gyro_variance, gyro_mean, noise_level, position_change)
        
        stats = {
            'acc_variance': float(acc_variance),
            'acc_mean': float(acc_mean),
            'acc_std': float(acc_std),
            'gyro_variance': float(gyro_variance),
            'gyro_mean': float(gyro_mean),
            'noise_level': float(noise_level),
            'position_change': float(position_change)
        }
        
        return activity, stats
    
    def classify_activity(self, acc_var, acc_mean, acc_std, gyro_var, gyro_mean, noise, pos_change):
        if acc_var < 0.5 and gyro_var < 0.001 and 8 < acc_mean < 12:
            return "STILL"
        if (0.5 < acc_var < 5.0 and 0.001 < gyro_var < 0.1 and 
            8 < acc_mean < 15 and 0.5 < acc_std < 2.5):
            return "WALKING"
        if (0.3 < acc_var < 3.0 and 0.01 < gyro_var < 0.3 and 
            9 < acc_mean < 16 and noise < 2.0):
            return "BIKING"
        if acc_var > 2.0 and noise > 1.5 and gyro_var > 0.02:
            return "SCOOTER"
        return "UNKNOWN"

# ----------------------------------------------------------------- 
# DATA COLLECTOR
# ----------------------------------------------------------------- 
class PhyphoxDataCollector:
    def __init__(self):
        self.latest_data = None
        self.classifier = MotionClassifier()
        self.dead_reckoning = AdvancedDeadReckoning()
        self.window_data = []
        self.acc_vert_history = []
        self.running = False
        self.thread = None
        
        self.stats = {
            'activity': 'COLLECTING DATA...',
            'motion_stats': {},
            'total_distance': 0.0,
            'num_steps': 0,
            'num_zupts': 0
        }
    
    def get_sensor_data(self):
        try:
            response = requests.get(FULL_REQUEST_URL, timeout=2)
            response.raise_for_status()
            json_data = response.json()
            
            data = {}
            for sensor in SENSORS_TO_GET:
                try:
                    data[sensor] = json_data["buffer"][sensor]["buffer"][-1]
                except (KeyError, IndexError, TypeError):
                    data[sensor] = None
            return data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def save_to_csv(self, data, activity, stats, dr_result):
        file_path = 'unified_motion_log.csv'
        file_exists = os.path.isfile(file_path)
        
        with open(file_path, mode='a', newline='') as file:
            fieldnames = ['timestamp', 'activity'] + SENSORS_TO_GET + list(stats.keys()) + \
                        ['pos_x', 'pos_y', 'pos_z', 'vel_x', 'vel_y', 'vel_z', 'is_stationary', 'step_detected']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            row_data = {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'activity': activity,
                **data,
                **stats,
                'pos_x': dr_result['position'][0],
                'pos_y': dr_result['position'][1],
                'pos_z': dr_result['position'][2],
                'vel_x': dr_result['velocity'][0],
                'vel_y': dr_result['velocity'][1],
                'vel_z': dr_result['velocity'][2],
                'is_stationary': dr_result['is_stationary'],
                'step_detected': dr_result['step_detected']
            }
            writer.writerow(row_data)
    
    def update_statistics(self):
        if len(self.dead_reckoning.history['position']) > 1:
            positions = np.array(self.dead_reckoning.history['position'])
            self.stats['total_distance'] = float(np.sum(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)))
            self.stats['num_steps'] = int(np.sum(self.dead_reckoning.history['steps_detected']))
            self.stats['num_zupts'] = int(np.sum(self.dead_reckoning.history['zupt_detected']))
    
    def collect_loop(self):
        while self.running:
            data = self.get_sensor_data()
            
            if data and all(data[s] is not None for s in ['accX', 'accY', 'accZ', 'gyrX', 'gyrY', 'gyrZ']):
                # Motion classification
                self.classifier.add_data(data)
                activity, motion_stats = self.classifier.analyze_motion()
                
                # Dead reckoning
                acc = np.array([data['accX'], data['accY'], data['accZ']])
                gyr = np.array([data['gyrX'], data['gyrY'], data['gyrZ']])
                
                self.window_data.append((acc, gyr))
                if len(self.window_data) > 20:
                    self.window_data.pop(0)
                
                self.acc_vert_history.append(np.linalg.norm(acc))
                if len(self.acc_vert_history) > 50:
                    self.acc_vert_history.pop(0)
                
                dr_result = self.dead_reckoning.process_measurement(acc, gyr, self.window_data, self.acc_vert_history)
                
                # Update stats
                self.update_statistics()
                self.stats['activity'] = activity
                self.stats['motion_stats'] = motion_stats
                
                # Store latest data
                self.latest_data = {
                    'sensor_data': data,
                    'activity': activity,
                    'motion_stats': motion_stats,
                    'dead_reckoning': {
                        'position': dr_result['position'].tolist(),
                        'velocity': dr_result['velocity'].tolist(),
                        'is_stationary': dr_result['is_stationary'],
                        'step_detected': dr_result['step_detected']
                    },
                    'statistics': self.stats.copy(),
                    'trajectory': [pos.tolist() for pos in self.dead_reckoning.history['position'][-100:]]
                }
                
                # Save to CSV
                self.save_to_csv(data, activity, motion_stats, dr_result)
            
            time.sleep(0.5)
    
    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self.collect_loop, daemon=True)
            self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

# ----------------------------------------------------------------- 
# FLASK API
# ----------------------------------------------------------------- 
app = Flask(__name__)
CORS(app)
collector = PhyphoxDataCollector()

@app.route('/api/data')
def get_data():
    if collector.latest_data:
        return jsonify(collector.latest_data)
    return jsonify({'error': 'No data available'}), 503

@app.route('/api/start')
def start_collection():
    collector.start()
    return jsonify({'status': 'started'})

@app.route('/api/stop')
def stop_collection():
    collector.stop()
    return jsonify({'status': 'stopped'})

@app.route('/api/status')
def get_status():
    return jsonify({
        'running': collector.running,
        'has_data': collector.latest_data is not None
    })

if __name__ == "__main__":
    print("🚀 Starting Unified IMU Motion Tracker Backend")
    print(f"📡 Phone URL: {PHONE_URL}")
    print("🌐 API running on http://localhost:5000")
    print("\nEndpoints:")
    print("  - GET /api/data    : Get latest sensor data")
    print("  - GET /api/start   : Start data collection")
    print("  - GET /api/stop    : Stop data collection")
    print("  - GET /api/status  : Get collection status")
    
    collector.start()
    app.run(debug=False, host='0.0.0.0', port=5000)
