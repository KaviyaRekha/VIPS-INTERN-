from flask import Flask, request, jsonify
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)

# -----------------------
# MongoDB Setup
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://shnee2505_db_user:yuvas@cluster0.pxwhd1y.mongodb.net/punchdb?retryWrites=true&w=majority")
DB_NAME = os.getenv("DB_NAME", "punchdb")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

punch_collection = db["punch_logs"]
leave_collection = db["leave_requests"]

print("✅ MongoDB Connected")

# -----------------------
# Punch In API
# -----------------------
@app.route("/api/punch", methods=["POST"])
def punch_in():
    data = request.json
    punch_data = {
        "employeeId": data.get("employeeId"),
        "password": data.get("password"),  # For demo only, hash in real projects
        "address": data.get("address"),
        "street": data.get("street"),
        "area": data.get("area"),
        "city": data.get("city"),
        "state": data.get("state"),
        "country": data.get("country"),
        "latitude": data.get("latitude"),
        "longitude": data.get("longitude"),
        "status": data.get("status"),
        "punchTime": datetime.utcnow()
    }
    punch_collection.insert_one(punch_data)
    return jsonify({"message": "Punch-in stored successfully"}), 201

# -----------------------
# Leave Apply API
# -----------------------
@app.route("/api/leave-apply", methods=["POST"])
def leave_apply():
    data = request.json
    leave_data = {
        "employeeId": data.get("employeeId"),
        "leaveType": data.get("leaveType"),
        "fromDate": data.get("fromDate"),
        "toDate": data.get("toDate"),
        "reason": data.get("reason"),
        "status": "Pending",
        "appliedAt": datetime.utcnow()
    }
    leave_collection.insert_one(leave_data)
    return jsonify({"message": "Leave applied successfully"}), 201

# -----------------------
# Attendance API
# -----------------------
@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    employeeId = request.args.get("employeeId")
    query = {}
    if employeeId:
        query["employeeId"] = employeeId
    punches = list(punch_collection.find(query, {"_id": 0}))
    return jsonify({"attendance": punches}), 200

# -----------------------
# Leave Requests API
# -----------------------
@app.route("/api/leaves", methods=["GET"])
def get_leaves():
    employeeId = request.args.get("employeeId")
    query = {}
    if employeeId:
        query["employeeId"] = employeeId
    leaves = list(leave_collection.find(query, {"_id": 0}))
    return jsonify({"leaves": leaves}), 200

# -----------------------
# Run Server
# -----------------------
if __name__ == "__main__":
    PORT = int(os.getenv("PORT", 8000))
    app.run(debug=True, port=PORT)
