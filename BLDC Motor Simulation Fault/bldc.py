"""
=================================================================================
BLDC MOTOR FAULT PREDICTION SYSTEM - COMPLETE WITH AUTO CIRCUIT GENERATOR
=================================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import serial
import time
import warnings
import webbrowser
import os
warnings.filterwarnings('ignore')

# =================================================================================
# SECTION 1: DATA GENERATION
# =================================================================================

def generate_training_data(n_samples=2000):
    """Generate synthetic BLDC motor sensor data for training"""
    np.random.seed(42)
    data = []
    
    for _ in range(int(n_samples * 0.5)):
        current = np.random.normal(2.5, 0.3)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(45, 5)
        vibration = np.random.normal(0.5, 0.1)
        speed = np.random.normal(1500, 50)
        data.append([current, voltage, temperature, vibration, speed, 0])
    
    for _ in range(int(n_samples * 0.15)):
        current = np.random.normal(3.5, 0.4)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(75, 8)
        vibration = np.random.normal(0.7, 0.15)
        speed = np.random.normal(1450, 80)
        data.append([current, voltage, temperature, vibration, speed, 1])
    
    for _ in range(int(n_samples * 0.15)):
        current = np.random.normal(2.8, 0.4)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(55, 7)
        vibration = np.random.normal(1.5, 0.3)
        speed = np.random.normal(1400, 100)
        data.append([current, voltage, temperature, vibration, speed, 2])
    
    for _ in range(int(n_samples * 0.1)):
        current = np.random.normal(3.0, 0.5)
        voltage = np.random.normal(9.5, 1.0)
        temperature = np.random.normal(50, 6)
        vibration = np.random.normal(0.8, 0.2)
        speed = np.random.normal(1200, 150)
        data.append([current, voltage, temperature, vibration, speed, 3])
    
    for _ in range(int(n_samples * 0.1)):
        current = np.random.normal(5.0, 0.6)
        voltage = np.random.normal(11.0, 0.5)
        temperature = np.random.normal(65, 8)
        vibration = np.random.normal(1.0, 0.25)
        speed = np.random.normal(1300, 120)
        data.append([current, voltage, temperature, vibration, speed, 4])
    
    df = pd.DataFrame(data, columns=['Current', 'Voltage', 'Temperature', 'Vibration', 'Speed', 'Fault'])
    return df

# =================================================================================
# SECTION 2: MODEL TRAINING
# =================================================================================

def train_model():
    """Train the machine learning model"""
    print("="*80)
    print("BLDC MOTOR FAULT PREDICTION - MODEL TRAINING")
    print("="*80)
    print("\nGenerating training data...")
    df = generate_training_data()
    
    print(f"\nDataset shape: {df.shape}")
    print("\nFault distribution:")
    fault_counts = df['Fault'].value_counts().sort_index()
    fault_names = ['Normal', 'Overheating', 'Bearing Fault', 'Voltage Fault', 'Overcurrent']
    for i, count in fault_counts.items():
        print(f"  {fault_names[i]}: {count} samples ({count/len(df)*100:.1f}%)")
    
    X = df.drop('Fault', axis=1)
    y = df['Fault']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print(f"MODEL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*80}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=fault_names, digits=3))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_names, yticklabels=fault_names)
    plt.title('BLDC Motor Fault Prediction - Confusion Matrix')
    plt.ylabel('Actual Fault Type')
    plt.xlabel('Predicted Fault Type')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f} {'█' * int(row['Importance'] * 50)}")
    
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'])
    plt.xlabel('Importance Score')
    plt.title('Feature Importance for Fault Prediction')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✓ Feature importance plot saved as 'feature_importance.png'")
    
    joblib.dump(model, 'bldc_fault_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✓ Model saved as 'bldc_fault_model.pkl'")
    print("✓ Scaler saved as 'scaler.pkl'")
    print("\n" + "="*80)
    
    return model, scaler

# =================================================================================
# SECTION 3: CIRCUIT DIAGRAM GENERATOR
# =================================================================================

def generate_html_circuit():
    """Generate interactive HTML circuit diagram"""
    
    html = """<!DOCTYPE html>
<html><head><meta charset="UTF-8">
<title>BLDC Circuit Diagram</title>
<style>
body{font-family:Arial;background:linear-gradient(135deg,#667eea,#764ba2);padding:20px;margin:0}
.container{max-width:1400px;margin:0 auto;background:#fff;border-radius:20px;box-shadow:0 20px 60px rgba(0,0,0,0.3)}
.header{background:linear-gradient(135deg,#667eea,#764ba2);color:#fff;padding:30px;text-align:center;border-radius:20px 20px 0 0}
h1{font-size:2.5em;margin:0 0 10px}
.content{padding:40px}
svg{width:100%;height:700px;border:3px solid #667eea;border-radius:15px;background:#f8f9fa}
.component{cursor:pointer;transition:all 0.3s}
.component:hover{filter:drop-shadow(0 0 10px rgba(102,126,234,0.8))}
.legend{margin-top:30px;padding:25px;background:#f8f9fa;border-radius:15px}
.legend-item{display:inline-block;margin:10px 20px}
.color-box{display:inline-block;width:40px;height:15px;margin-right:10px;border-radius:3px}
.info{margin-top:20px;padding:20px;background:#e8f0ff;border-radius:10px;border-left:5px solid #667eea}
</style></head><body>
<div class="container">
<div class="header"><h1>🔧 BLDC Motor Circuit</h1><p>ESP32 Fault Detection System</p></div>
<div class="content">
<svg viewBox="0 0 1200 700">
<rect x="450" y="250" width="120" height="200" fill="#2c3e50" stroke="#000" stroke-width="2" rx="5"/>
<text x="510" y="240" font-size="16" font-weight="bold" text-anchor="middle">ESP32</text>
<circle cx="460" cy="280" r="4" fill="#FFD700"/><text x="435" y="285" font-size="11" text-anchor="end">GPIO34</text>
<circle cx="460" cy="310" r="4" fill="#FFD700"/><text x="435" y="315" font-size="11" text-anchor="end">GPIO35</text>
<circle cx="560" cy="280" r="4" fill="#FFD700"/><text x="585" y="285" font-size="11">GPIO26</text>
<circle cx="560" cy="310" r="4" fill="#FFD700"/><text x="585" y="315" font-size="11">GPIO13</text>
<circle cx="560" cy="340" r="4" fill="#FFD700"/><text x="585" y="345" font-size="11">GPIO12</text>
<circle cx="510" cy="250" r="4" fill="#000"/><text x="510" y="235" font-size="11" text-anchor="middle">GND</text>
<circle cx="510" cy="460" r="4" fill="#e74c3c"/><text x="510" y="480" font-size="11" text-anchor="middle">3.3V</text>
<circle cx="250" cy="280" r="30" fill="#95a5a6" stroke="#000" stroke-width="2"/>
<line x1="250" y1="250" x2="250" y2="230" stroke="#000" stroke-width="3"/>
<text x="250" y="200" font-size="14" font-weight="bold" text-anchor="middle">POT</text>
<circle cx="750" cy="300" r="50" fill="#e74c3c" stroke="#000" stroke-width="3"/>
<text x="750" y="310" fill="#fff" font-size="20" font-weight="bold" text-anchor="middle">M</text>
<text x="750" y="380" font-size="14" font-weight="bold" text-anchor="middle">Motor</text>
<circle cx="750" cy="500" r="15" fill="#27ae60" stroke="#000" stroke-width="2"/>
<text x="750" y="540" font-size="12" text-anchor="middle">LED OK</text>
<circle cx="850" cy="500" r="15" fill="#e74c3c" stroke="#000" stroke-width="2"/>
<text x="850" y="540" font-size="12" text-anchor="middle">LED Fault</text>
<rect x="640" y="430" width="60" height="12" fill="#d4a574" stroke="#000"/>
<text x="670" y="425" font-size="10" text-anchor="middle">220Ω</text>
<rect x="740" y="430" width="60" height="12" fill="#d4a574" stroke="#000"/>
<text x="770" y="425" font-size="10" text-anchor="middle">220Ω</text>
<path d="M 250,230 L 250,200 L 450,200 L 450,460 L 506,460" stroke="#e74c3c" stroke-width="3" fill="none"/>
<path d="M 510,254 L 510,600 L 200,600 L 200,320 L 220,300" stroke="#000" stroke-width="3" fill="none"/>
<path d="M 280,300 L 320,280 L 456,280" stroke="#27ae60" stroke-width="2" fill="none"/>
<path d="M 564,280 L 600,280 L 600,250 L 750,250" stroke="#e74c3c" stroke-width="3" fill="none"/>
<path d="M 750,350 L 750,600" stroke="#000" stroke-width="3" fill="none"/>
<path d="M 564,310 L 640,310 L 640,430" stroke="#27ae60" stroke-width="2" fill="none"/>
<path d="M 700,436 L 750,485" stroke="#27ae60" stroke-width="2" fill="none"/>
<path d="M 564,340 L 740,340 L 740,430" stroke="#e74c3c" stroke-width="2" fill="none"/>
<path d="M 800,436 L 850,485" stroke="#e74c3c" stroke-width="2" fill="none"/>
<path d="M 750,515 L 750,600" stroke="#000" stroke-width="3" fill="none"/>
<path d="M 850,515 L 850,600 L 750,600" stroke="#000" stroke-width="3" fill="none"/>
</svg>
<div class="legend"><h3>Legend</h3>
<div class="legend-item"><span class="color-box" style="background:#e74c3c"></span>Power</div>
<div class="legend-item"><span class="color-box" style="background:#000"></span>Ground</div>
<div class="legend-item"><span class="color-box" style="background:#27ae60"></span>Signal</div>
</div>
<div class="info"><h3>Pin Connections</h3>
<p><b>GPIO34:</b> Potentiometer (Speed) | <b>GPIO26:</b> Motor PWM<br>
<b>GPIO13:</b> Green LED (Normal) | <b>GPIO12:</b> Red LED (Fault)</p></div>
</div></div></body></html>"""
    
    with open('bldc_circuit.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print("\n✓ Circuit diagram saved: bldc_circuit.html")
    return 'bldc_circuit.html'

def generate_arduino_code():
    """Generate Arduino code file"""
    code = """// BLDC Motor Fault Detection - ESP32
#define POT_PIN 34
#define MOTOR_PWM_PIN 26
#define LED_NORMAL 13
#define LED_FAULT 12

const int pwmFreq = 5000;
const int pwmChannel = 0;
const int pwmResolution = 8;

float motorCurrent = 0, motorVoltage = 0, motorTemp = 0;
float motorVibration = 0;
int motorSpeed = 0;
bool faultMode = false;
int faultType = 0;

void setup() {
  Serial.begin(115200);
  ledcSetup(pwmChannel, pwmFreq, pwmResolution);
  ledcAttachPin(MOTOR_PWM_PIN, pwmChannel);
  pinMode(LED_NORMAL, OUTPUT);
  pinMode(LED_FAULT, OUTPUT);
  digitalWrite(LED_NORMAL, HIGH);
  Serial.println("BLDC Fault Detection Ready");
}

void loop() {
  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\\n');
    if (cmd.startsWith("f")) {
      faultType = cmd.substring(1).toInt();
      faultMode = (faultType > 0);
    }
  }
  
  int pot = analogRead(POT_PIN);
  int pwm = map(pot, 0, 4095, 0, 255);
  ledcWrite(pwmChannel, pwm);
  motorSpeed = map(pwm, 0, 255, 0, 2000);
  
  float sf = motorSpeed / 2000.0;
  if (faultMode) {
    if (faultType == 1) { motorCurrent=3.5; motorTemp=75; motorVibration=0.7; }
    else if (faultType == 2) { motorCurrent=2.8; motorTemp=55; motorVibration=1.5; }
    else if (faultType == 3) { motorCurrent=3.0; motorVoltage=9.5; motorTemp=50; }
    else { motorCurrent=5.0; motorTemp=65; motorVibration=1.0; }
    motorVoltage = 12.0;
  } else {
    motorCurrent = 1.5 + sf * 1.5;
    motorVoltage = 11.8;
    motorTemp = 40 + sf * 15;
    motorVibration = 0.3 + sf * 0.3;
  }
  
  Serial.print("DATA:");
  Serial.print(motorCurrent,2); Serial.print(",");
  Serial.print(motorVoltage,2); Serial.print(",");
  Serial.print(motorTemp,1); Serial.print(",");
  Serial.print(motorVibration,2); Serial.print(",");
  Serial.println(motorSpeed);
  
  bool fault = (motorTemp>70 || motorCurrent>4.0 || motorVibration>1.2);
  digitalWrite(LED_NORMAL, !fault);
  digitalWrite(LED_FAULT, fault);
  delay(1000);
}"""
    
    with open('bldc_esp32.ino', 'w', encoding='utf-8') as f:
        f.write(code)
    
    print("✓ Arduino code saved: bldc_esp32.ino")

# =================================================================================
# SECTION 4: SERIAL PREDICTION
# =================================================================================

def predict_from_serial(port='COM3', baudrate=115200):
    """Real-time prediction from ESP32"""
    try:
        model = joblib.load('bldc_fault_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except:
        print("Model not found. Training...")
        model, scaler = train_model()
    
    fault_names = ['Normal', 'Overheating', 'Bearing Fault', 'Voltage Fault', 'Overcurrent']
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"\n✓ Connected to {port}")
        print("Waiting for data...\n")
        time.sleep(2)
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith("DATA:"):
                    try:
                        vals = line.replace("DATA:", "").split(',')
                        current, voltage, temp, vib, speed = map(float, vals)
                        
                        input_data = np.array([[current, voltage, temp, vib, speed]])
                        input_scaled = scaler.transform(input_data)
                        
                        pred = model.predict(input_scaled)[0]
                        probs = model.predict_proba(input_scaled)[0]
                        
                        print("="*70)
                        print(f"Current:{current:.2f}A Voltage:{voltage:.2f}V Temp:{temp:.1f}°C Vib:{vib:.2f}g Speed:{speed:.0f}RPM")
                        print(f"PREDICTION: {fault_names[pred]} (Confidence: {probs[pred]*100:.1f}%)")
                        print("="*70 + "\n")
                        
                    except Exception as e:
                        print(f"Error: {e}")
                        
    except serial.SerialException:
        print(f"Cannot open {port}. Running simulation...")
        simulate_predictions(model, scaler)

# =================================================================================
# SECTION 5: SIMULATION
# =================================================================================

def simulate_predictions(model, scaler):
    """Simulation mode"""
    fault_names = ['Normal', 'Overheating', 'Bearing Fault', 'Voltage Fault', 'Overcurrent']
    
    scenarios = [
        ([2.5, 12.0, 45, 0.5, 1500], "Normal"),
        ([3.5, 12.0, 78, 0.7, 1450], "Overheating"),
        ([2.8, 12.0, 55, 1.8, 1400], "Bearing Fault"),
        ([3.0, 9.0, 50, 0.8, 1200], "Voltage Fault"),
        ([5.2, 11.0, 65, 1.0, 1300], "Overcurrent"),
    ]
    
    for vals, name in scenarios:
        input_data = np.array([vals])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]
        
        print(f"\n{'='*70}")
        print(f"Scenario: {name}")
        print(f"Current:{vals[0]:.2f}A Voltage:{vals[1]:.2f}V Temp:{vals[2]:.1f}°C")
        print(f"PREDICTION: {fault_names[pred]} ({probs[pred]*100:.1f}%)")
        print("="*70)
        time.sleep(1)

# =================================================================================
# MAIN MENU
# =================================================================================

def main():
    print("\n" + "="*80)
    print("🔧 BLDC MOTOR FAULT PREDICTION SYSTEM")
    print("="*80)
    
    while True:
        print("\n1. Train Model")
        print("2. Connect ESP32 (Real-time)")
        print("3. Run Simulation")
        print("4. Generate Circuit Diagram (HTML)")
        print("5. Generate Arduino Code")
        print("6. Generate ALL Files")
        print("7. Exit")
        
        choice = input("\nChoice: ").strip()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            port = input("COM port (e.g., COM3): ").strip() or 'COM3'
            predict_from_serial(port)
        elif choice == '3':
            try:
                model = joblib.load('bldc_fault_model.pkl')
                scaler = joblib.load('scaler.pkl')
            except:
                model, scaler = train_model()
            simulate_predictions(model, scaler)
        elif choice == '4':
            html_file = generate_html_circuit()
            webbrowser.open('file://' + os.path.abspath(html_file))
        elif choice == '5':
            generate_arduino_code()
        elif choice == '6':
            print("\nGenerating all files...")
            train_model()
            html_file = generate_html_circuit()
            generate_arduino_code()
            print("\n✓ All files generated!")
            webbrowser.open('file://' + os.path.abspath(html_file))
        elif choice == '7':
            break

if __name__ == "__main__":
    main()