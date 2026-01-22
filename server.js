const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
const dotenv = require("dotenv");
const bcrypt = require("bcryptjs");

dotenv.config();

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static('.'));

// ---------------- MongoDB Connection ----------------
mongoose.connect(process.env.MONGO_URI)
.then(() => console.log("✅ MongoDB connected"))
.catch(err => console.error("❌ MongoDB error:", err));

// ================== EMPLOYEE MASTER ==================
const EmployeeSchema = new mongoose.Schema({
    employeeId: { type: String, required: true, unique: true },
    password: { type: String, required: true },
    name: { type: String },
    createdAt: { type: Date, default: Date.now }
});

EmployeeSchema.index({ employeeId: 1 });

const Employee = mongoose.model("Employee", EmployeeSchema);

// ================== PUNCH DETAILS ==================
const PunchSchema = new mongoose.Schema({
    employeeId: { type: String, required: true },
    address: String,
    street: String,
    area: String,
    city: String,
    state: String,
    country: String,
    latitude: Number,
    longitude: Number,
    status: { type: String, enum: ["IN", "OUT"], required: true },
    punchedAt: { type: Date, default: Date.now }
});

const Punch = mongoose.model("Punch", PunchSchema);

// ================== API 1: REGISTER EMPLOYEE ==================
app.post("/api/employee/register", async (req, res) => {
    try {
        const { employeeId, password, name } = req.body;

        if (!employeeId || !password) {
            return res.json({ success: false, message: "Employee ID & password required" });
        }

        const existing = await Employee.findOne({ employeeId });

        if (existing) {
            return res.json({ success: false, message: "Employee already exists" });
        }

        const hashed = await bcrypt.hash(password, 10);

        const emp = new Employee({
            employeeId,
            password: hashed,
            name
        });

        await emp.save();

        res.json({ success: true, message: "Employee registered successfully" });

    } catch (err) {
        console.error(err);
        res.status(500).json({ success: false, message: "Server error" });
    }
});

// ================== API 2: EMPLOYEE LOGIN ==================
app.post("/api/employee/login", async (req, res) => {
    try {
        const { employeeId, password } = req.body;

        const emp = await Employee.findOne({ employeeId });

        if (!emp) {
            return res.json({ success: false, message: "Invalid employee ID" });
        }

        const match = await bcrypt.compare(password, emp.password);

        if (!match) {
            return res.json({ success: false, message: "Invalid password" });
        }

        res.json({
            success: true,
            message: "Login successful",
            employee: {
                employeeId: emp.employeeId,
                name: emp.name
            }
        });

    } catch (error) {
        console.error(error);
        res.status(500).json({ success: false, message: "Server error" });
    }
});

// ================== API 3: SAVE PUNCH ==================
app.post("/api/punch", async (req, res) => {
    try {
        const { employeeId, status } = req.body;

        if (!employeeId || !status) {
            return res.json({ success: false, message: "Employee ID & status required" });
        }

        const punch = new Punch(req.body);
        await punch.save();

        res.json({ success: true, message: "Punch saved successfully" });

    } catch (error) {
        console.error(error);
        res.status(500).json({ success: false, message: "Server error" });
    }
});

// ================== START SERVER ==================
const PORT = process.env.PORT || 3000;
app.get('/', (req, res) => res.sendFile(__dirname + '/index1.html'));
app.listen(PORT, () => console.log(`🚀 Server running on port ${PORT}`));



