from flask import Flask, request, jsonify, send_from_directory
import time
import requests
import json
import os
from datetime import datetime
import logging

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv

# ---------------- LOGGING ----------------
logging.getLogger("werkzeug").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("ml_api")

# ---------------- ENV ----------------
load_dotenv()

app = Flask(__name__, static_folder="static")

FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")

# ---------------- NORMALIZATION ----------------
# Small values friendly (college project)

def normalize_distance(distance, safe_radius):
    if safe_radius <= 0:
        return 0
    return min(distance / (safe_radius * 2), 1.0)

def normalize_time(time_outside):
    # 30 minutes → full risk
    return min(time_outside / 30.0, 1.0)

def normalize_speed(speed, avg_speed):
    if avg_speed <= 0:
        return 0.0
    return min(speed / (avg_speed * 1.5), 1.0)

# ---------------- LEARNING ----------------
def update_learning(old_avg, new_value, count):
    return (old_avg * count + new_value) / (count + 1)

# ---------------- WEIGHTS ----------------
def normalize_weights(weights):
    total = sum(weights.values())
    if total <= 0:
        return {"distance": 0.4, "time": 0.4, "speed": 0.2}
    return {k: round(v / total, 3) for k, v in weights.items()}

def update_weights(weights, distance, time_outside, speed):
    lr = 0.015
    weights["distance"] += lr * (distance / 10)
    weights["time"] += lr * (time_outside / 10)
    weights["speed"] += lr * (speed / 2)

    # floor to avoid collapse
    for k in weights:
        weights[k] = max(weights[k], 0.05)

    return normalize_weights(weights)

# ---------------- CONTEXT ----------------
def night_time_boost():
    hour = datetime.now().hour
    if hour >= 22 or hour < 5:
        return 0.15
    return 0.0

def context_boost(battery, is_raining, light_level):
    boost = 0.0
    if battery <= 20:
        boost += 0.10
    if is_raining:
        boost += 0.10
    if light_level == "low":
        boost += 0.08
    return boost

# ---------------- TREND ----------------
def risk_trend_boost(history, current):
    if not history or len(history) < 2:
        return 0.0
    last = list(history.values())[-1]
    delta = current - last
    if delta > 0.3:
        return 0.15
    return 0.0

def update_risk_history(history, risk):
    history = history or {}
    history[str(int(time.time()))] = risk
    return dict(list(history.items())[-5:])

# ---------------- ZONE ----------------
def get_zone_key(lat, lon, precision=3):
    return f"{round(lat, precision)}_{round(lon, precision)}"

def update_zone_heatmap(zone_map, key):
    zone_map = zone_map or {}
    zone_map[key] = zone_map.get(key, 0) + 1
    return dict(list(zone_map.items())[-50:])

def zone_risk_boost(zone_map, key):
    if not zone_map or key not in zone_map:
        return 0.05
    visits = zone_map[key]
    return min(visits * 0.01, 0.15)

# ---------------- FEEDBACK ----------------
def apply_feedback(patient_id, weights, feedback):
    adjust = 0.05
    if feedback == "false_alarm":
        weights["distance"] -= adjust
        weights["time"] -= adjust
    elif feedback == "correct_alert":
        weights["distance"] += adjust
        weights["time"] += adjust

    weights = normalize_weights(weights)

    # clear feedback after use
    requests.patch(
        f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
        json={"lastFeedback": None}
    )

    return weights

# ---------------- RISK CORE ----------------
def calculate_risk(distance, time_outside, speed, safe_radius,
                   avg_speed, weights, battery, is_raining,
                   light_level, history, zone_map, zone_key):

    nd = normalize_distance(distance, safe_radius)
    nt = normalize_time(time_outside)
    ns = normalize_speed(speed, avg_speed)

    base = (
        weights["distance"] * nd +
        weights["time"] * nt +
        weights["speed"] * ns
    )

    total = base
    total += night_time_boost()
    total += context_boost(battery, is_raining, light_level)
    total += risk_trend_boost(history, base)
    total += zone_risk_boost(zone_map, zone_key)

    logger.info(
        f"Inputs | distance={distance}, safe_radius={safe_radius}, "
        f"time={time_outside}, speed={speed}"
    )

    return round(min(total, 1.0), 2)

# ---------------- FCM (SAFE) ----------------
def get_access_token():
    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds:
        raise RuntimeError("FCM credentials missing")
    sa = json.loads(creds)
    cred = service_account.Credentials.from_service_account_info(
        sa, scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    cred.refresh(Request())
    return cred.token

def send_push(token, title, body, patient_id=None):
    try:
        access_token = get_access_token()

        response = requests.post(
            f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            },
            json={
                "message": {
                    "token": token,
                    "notification": {
                        "title": title,
                        "body": body
                    },
                    "android": {
                        "priority": "HIGH"
                    }
                }
            },
            timeout=5
        )

        if response.status_code == 404:
            if "UNREGISTERED" in response.text and patient_id:
                # delete bad token
                requests.patch(
                    f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
                    json={"fcmToken": None}
                )
                logger.warning("Deleted invalid FCM token")

        logger.info(response.text)

    except Exception as e:
        logger.error(f"Push failed: {e}")

def should_send_alert(prev, new):
    return prev != "alert" and new == "alert"

# ---------------- API ----------------

@app.route("/update-fcm-token", methods=["POST"])
def update_fcm_token():
    data = request.get_json()

    patient_id = data.get("patientId")
    token = data.get("fcmToken")

    if not patient_id or not token:
        return jsonify({"status": "error"}), 400

    requests.patch(
        f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
        json={"fcmToken": token}
    )

    logger.info(f"Token updated for {patient_id}")

    return jsonify({"status": "success"})

@app.route("/firebase-config")
def firebase_config():
    return jsonify({
        "apiKey": os.getenv("FIREBASE_API_KEY"),
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "databaseURL": os.getenv("FIREBASE_DB_URL"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID"),
    })

@app.route("/test-firebase", methods=["GET"])
def test_firebase():
    """Test Firebase connection and permissions"""
    logger.info("Testing Firebase connection...")
    
    if not FIREBASE_DB_URL:
        return jsonify({"error": "FIREBASE_DB_URL not set"}), 500
    
    logger.info(f"Firebase DB URL: {FIREBASE_DB_URL}")
    
    try:
        # Test GET
        test_url = f"{FIREBASE_DB_URL}/.json"
        logger.info(f"Testing GET: {test_url}")
        resp = requests.get(test_url, timeout=5)
        logger.info(f"GET Response: {resp.status_code} - {resp.text[:200]}")
        
        if resp.status_code == 401:
            return jsonify({
                "error": "Unauthorized - Firebase requires authentication",
                "status": resp.status_code,
                "message": "Check database rules and credentials"
            }), 401
        
        return jsonify({
            "message": "Firebase connection OK",
            "status": resp.status_code,
            "firebase_db_url": FIREBASE_DB_URL
        })
        
    except Exception as e:
        logger.error(f"Firebase connection error: {str(e)}", exc_info=True)
        return jsonify({
            "error": str(e),
            "firebase_db_url": FIREBASE_DB_URL
        }), 500

@app.route("/predict", methods=["POST"])
def predict():
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        return jsonify({"riskScore": 0, "riskLevel": "normal"})

    patient_id = d.get("patientId")
    if not patient_id:
        return jsonify({"riskScore": 0, "riskLevel": "normal"})

    speed = float(d.get("speed", 0))
    distance = float(d.get("distance", 0))
    time_outside = int(d.get("time_outside", 0))
    
    logger.info(f"Received request: patientId={patient_id}, distance={distance}, speed={speed}")
    lat = float(d.get("latitude", 0))
    lon = float(d.get("longitude", 0))

    battery = int(d.get("battery", 100))
    is_raining = bool(d.get("isRaining", False))
    light = d.get("lightLevel", "normal")
    feedback = d.get("feedback")

    resp = requests.get(
        f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
        timeout=5
    )

    try:
        patient = resp.json() if resp.status_code == 200 else None
    except Exception:
        patient = None

    if not patient:
        return jsonify({"riskScore": 0, "riskLevel": "normal"})

    learning = patient.get("learning", {})
    weights = learning.get(
        "weights", {"distance": 0.4, "time": 0.4, "speed": 0.2}
    )
    weights = normalize_weights(weights)

    avg_speed = learning.get("avgSpeed", max(speed, 0.1))
    samples = learning.get("samples", 0)
    safe_radius = max(float(patient.get("safeRadius", 5)), 1)

    zone_map = patient.get("zoneHeatmap", {})
    history = patient.get("riskHistory", {})
    zone_key = get_zone_key(lat, lon)

    zone_map = update_zone_heatmap(zone_map, zone_key)
    weights = update_weights(weights, distance, time_outside, speed)

    if feedback:
        weights = apply_feedback(patient_id, weights, feedback)

    risk = calculate_risk(
        distance, time_outside, speed, safe_radius,
        avg_speed, weights, battery, is_raining,
        light, history, zone_map, zone_key
    )

    history = update_risk_history(history, risk)

    level = "normal"
    if risk > 0.8:
        level = "alert"
    elif risk > 0.6:
        level = "warning"

    status = "inside"
    if distance > safe_radius:
       status = "outside"

    # Update main patient data including distance
    payload = {
        # movement info
        "currentLocation": {
            "lat": lat,
            "lon": lon
        },
        "speed": speed,
        "distance": distance,
        "status": status,
        "riskScore": risk,
        "riskLevel": level,
        "lastUpdated": int(time.time() * 1000),

        # learning system
        "learning": {
            "avgSpeed": update_learning(avg_speed, speed, samples),
            "samples": samples + 1,
            "weights": weights,
            "lastUpdated": int(time.time())
        },

        # history
        "zoneHeatmap": zone_map,
        "riskHistory": history
    }
    
    logger.info(f"Firebase URL: {FIREBASE_DB_URL}")
    logger.info(f"Sending payload: {json.dumps(payload, default=str)}")
    
    try:
        url = f"{FIREBASE_DB_URL}/patients/{patient_id}.json"
        resp = requests.patch(url, json=payload, timeout=5)
        
        logger.info(f"Firebase Response Status: {resp.status_code}")
        logger.info(f"Firebase Response: {resp.text}")
        
        if resp.status_code not in [200, 204]:
            logger.error(f"❌ Firebase update FAILED: {resp.status_code} - {resp.text}")
        else:
            logger.info(f"✅ Distance updated: {distance}m for patient {patient_id}")
            
    except Exception as e:
        logger.error(f"❌ Firebase PATCH Exception: {str(e)}", exc_info=True)

    if level == "alert" and patient.get("fcmToken"):
        send_push(
            patient["fcmToken"],
            "🚨 Wandering Alert",
            f"High risk detected ({risk})",
            patient_id
        )
    
    # Store alert for caretaker dashboard
    if level == "alert" and should_send_alert(d.get("prevRiskLevel"), level):
        try:
            requests.post(
                f"{FIREBASE_DB_URL}/alerts.json",
                json={
                    "patientId": patient_id,
                    "riskScore": risk,
                    "riskLevel": level,
                    "timestamp": int(time.time() * 1000),
                    "active": True,
                    "acknowledged": False,
                    "snoozedUntil": None
                },
                timeout=5
            )
            logger.info(f"Alert stored for patient {patient_id}")
        except Exception as e:
            logger.error(f"Failed to store alert: {e}")


    logger.info(
        f"Prediction OK | patient={patient_id} | level={level} | risk={risk}"
    )

    return jsonify({
        "riskScore": risk,
        "riskLevel": level,
        "triggerAlert": should_send_alert(
            d.get("prevRiskLevel"), level
        )
    })

@app.route("/")
def caretaker_dashboard():
    return send_from_directory("static", "caretaker.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
