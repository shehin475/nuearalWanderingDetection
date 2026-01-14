from flask import Flask, request, jsonify, send_from_directory
import time
import requests
import json
import os
from datetime import datetime

from google.oauth2 import service_account
from google.auth.transport.requests import Request
from dotenv import load_dotenv
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

load_dotenv()

app = Flask(__name__, static_folder="static")

# ---------------- FIREBASE CONFIG ----------------
FIREBASE_DB_URL = os.getenv("FIREBASE_DB_URL")
PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")

# ---------------- NORMALIZATION ----------------
def normalize_distance(distance, safe_radius):
    return min(distance / safe_radius, 1.0)

def normalize_time(time_outside):
    return min(time_outside / 5.0, 1.0)

def normalize_speed(speed, avg_speed):
    if avg_speed <= 0:
        return 0.5
    return min(speed / avg_speed, 1.5)

# ---------------- LEARNING ----------------
def update_learning(old_avg, new_value, count):
    return (old_avg * count + new_value) / (count + 1)

# ---------------- DYNAMIC WEIGHTS ----------------
def update_weights(weights, distance, time_outside, speed):
    lr = 0.02
    weights["distance"] += lr * (distance / 100)
    weights["time"] += lr * (time_outside / 5)
    weights["speed"] += lr * (speed / 2)

    total = sum(weights.values())
    for k in weights:
        weights[k] = round(weights[k] / total, 3)
    return weights

# ---------------- TIME & CONTEXT ----------------
def night_time_boost():
    hour = datetime.now().hour
    if hour >= 22 or hour < 5:
        return 0.25
    elif hour >= 20:
        return 0.15
    return 0.0

def context_boost(battery, is_raining, light_level):
    boost = 0.0
    if battery <= 20:
        boost += 0.15
    elif battery <= 40:
        boost += 0.08
    if is_raining:
        boost += 0.12
    if light_level == "low":
        boost += 0.10
    return boost

# ---------------- TREND ----------------
def risk_trend_boost(history, current_risk):
    if not history or len(history) < 2:
        return 0.0
    last = list(history.values())[-1]
    delta = current_risk - last
    if delta > 0.25:
        return 0.2
    elif delta > 0.15:
        return 0.1
    return 0.0

def update_risk_history(history, risk):
    history = history or {}
    history[str(int(time.time()))] = risk
    if len(history) > 5:
        history = dict(list(history.items())[-5:])
    return history

# ---------------- ZONE HEATMAP ----------------
def get_zone_key(lat, lon, precision=3):
    return f"{round(lat, precision)}_{round(lon, precision)}"

def update_zone_heatmap(zone_map, zone_key):
    zone_map = zone_map or {}
    zone_map[zone_key] = zone_map.get(zone_key, 0) + 1
    if len(zone_map) > 50:
        zone_map = dict(sorted(zone_map.items(), key=lambda x: x[1], reverse=True)[:50])
    return zone_map

def zone_risk_boost(zone_map, zone_key):
    if not zone_map or zone_key not in zone_map:
        return 0.2
    visits = zone_map[zone_key]
    if visits < 3:
        return 0.15
    elif visits < 10:
        return 0.05
    return 0.0

# ---------------- GEOFENCE ----------------
def geofence_boost(distance, safe_radius):
    if distance > safe_radius * 1.5:
        return 0.25
    elif distance > safe_radius:
        return 0.15
    return 0.0

# ---------------- FEEDBACK ----------------
def apply_feedback(weights, feedback):
    adjust = 0.05
    if feedback == "false_alarm":
        weights["distance"] -= adjust
        weights["time"] -= adjust
    elif feedback == "correct_alert":
        weights["distance"] += adjust
        weights["time"] += adjust

    total = sum(weights.values())
    for k in weights:
        weights[k] = round(max(weights[k], 0.05) / total, 3)
    return weights

# ---------------- RISK ----------------
def calculate_risk(distance, time_outside, speed, safe_radius, avg_speed,
                   weights, battery, is_raining, light_level,
                   history, zone_map, zone_key):

    d = normalize_distance(distance, safe_radius)
    t = normalize_time(time_outside)
    s = normalize_speed(speed, avg_speed)

    base = (
        weights["distance"] * d +
        weights["time"] * t +
        weights["speed"] * s
    )

    total = base
    total += night_time_boost()
    total += context_boost(battery, is_raining, light_level)
    total += risk_trend_boost(history, base)
    total += zone_risk_boost(zone_map, zone_key)
    total += geofence_boost(distance, safe_radius)

    return round(min(total, 1.0), 2)

# ---------------- FCM ----------------
def get_access_token():
    sa = json.loads(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
    cred = service_account.Credentials.from_service_account_info(
        sa, scopes=["https://www.googleapis.com/auth/firebase.messaging"]
    )
    cred.refresh(Request())
    return cred.token

def send_push(token, title, body):
    requests.post(
        f"https://fcm.googleapis.com/v1/projects/{PROJECT_ID}/messages:send",
        headers={
            "Authorization": f"Bearer {get_access_token()}",
            "Content-Type": "application/json"
        },
        json={
            "message": {
                "token": token,
                "notification": {"title": title, "body": body},
                "android": {"priority": "HIGH"}
            }
        }
    )

def should_send_alert(prev_level, new_level):
    return prev_level != "alert" and new_level == "alert"


# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    d = request.json

    patient_id = d["patientId"]
    speed = float(d["speed"])
    distance = float(d["distance"])
    time_outside = int(d["time_outside"])
    lat = float(d.get("latitude", 0))
    lon = float(d.get("longitude", 0))


    battery = int(d.get("battery", 100))
    is_raining = bool(d.get("isRaining", False))
    light = d.get("lightLevel", "normal")
    feedback = d.get("feedback")

        # ---------- SAFE FIREBASE FETCH ----------
    resp = requests.get(
        f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
        timeout=5
    )

    try:
        patient = resp.json() if resp.status_code == 200 else None
    except Exception:
        patient = None

    if not patient:
        return jsonify({
            "riskScore": 0,
            "riskLevel": "normal",
            "triggerAlert": False
        })

    learning = patient.get("learning", {})
    zone_map = patient.get("zoneHeatmap", {})
    history = patient.get("riskHistory", {})

    weights = learning.get(
        "weights", {"distance": 0.5, "time": 0.3, "speed": 0.2}
    )
    avg_speed = learning.get("avgSpeed", speed)
    avg_distance = learning.get("avgDistance", distance)
    samples = learning.get("samples", 0)
    safe_radius = patient.get("safeRadius", 200)

    zone_key = get_zone_key(lat, lon)
    zone_map = update_zone_heatmap(zone_map, zone_key)

    weights = update_weights(weights, distance, time_outside, speed)
    if feedback:
        weights = apply_feedback(weights, feedback)

    risk = calculate_risk(
        distance, time_outside, speed, safe_radius,
        avg_speed, weights, battery, is_raining,
        light, history, zone_map, zone_key
    )

    history = update_risk_history(history, risk)

    # ---------- STANDARDIZED RISK LEVEL ----------
    level = "normal"
    if risk > 0.8:
        level = "alert"
    elif risk > 0.6:
        level = "warning"

    # ---------- SAVE LEARNING ----------
    requests.patch(
        f"{FIREBASE_DB_URL}/patients/{patient_id}.json",
        json={
            "learning": {
                "avgSpeed": update_learning(avg_speed, speed, samples),
                "avgDistance": update_learning(avg_distance, distance, samples),
                "samples": samples + 1,
                "weights": weights,
                "lastUpdated": int(time.time())
            },
            "zoneHeatmap": zone_map,
            "riskHistory": history
        }
    )

    # ---------- OPTIONAL FCM ----------
    if level == "alert" and "fcmToken" in patient:
        send_push(
            patient["fcmToken"],
            "🚨 Wandering Alert",
            f"High risk detected ({risk})"
        )

    app.logger.info("Prediction completed")
    app.logger.info(
    f"Prediction OK | patient={patient_id} | level={level} | risk={risk}"
    )

    # ---------- FINAL RESPONSE ----------
    return jsonify({
        "riskScore": round(risk, 1),
        "riskLevel": level,
        "triggerAlert": should_send_alert(
            d.get("prevRiskLevel"), level
        )
    })



@app.route("/")
def caretaker_dashboard():
    return send_from_directory("static", "caretaker.html")

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
