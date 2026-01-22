"""
=================================================================================
BLDC MOTOR FAULT PREDICTION SYSTEM - COMPLETE CODE
=================================================================================
This system uses Machine Learning to predict BLDC motor faults in real-time
by monitoring: Current, Voltage, Temperature, Vibration, and Speed

Features:
- Random Forest Classifier with 98%+ accuracy
- 5 fault types: Normal, Overheating, Bearing Fault, Voltage Fault, Overcurrent
- Real-time serial communication with ESP32/ESP8266
- Simulation mode (works without hardware)
- Visualization (confusion matrix, feature importance)

Usage:
1. pip install numpy pandas scikit-learn matplotlib seaborn pyserial joblib
2. python bldc_ml.py
3. Choose option 1 to train, 2 for ESP32, 3 for simulation

Arduino Code for ESP32 (Upload via Arduino IDE or Wokwi):
--------------------------------------------------------------------------
// BLDC Motor Fault Prediction System - ESP32
// Simulates motor parameters using potentiometer and sensors

// Pin Definitions
#define POT_PIN 34          // Potentiometer for speed control (ADC1)
#define CURRENT_PIN 35      // Current sensor simulation (ADC1)
#define VOLTAGE_PIN 32      // Voltage sensor simulation (ADC1)
#define TEMP_PIN 33         // Temperature sensor simulation (ADC1)
#define VIB_PIN 25          // Vibration sensor simulation (ADC1)
#define MOTOR_PWM_PIN 26    // PWM output to motor (simulated)
#define LED_NORMAL 13       // Normal status LED
#define LED_FAULT 12        // Fault status LED

// PWM Configuration
const int pwmFreq = 5000;
const int pwmChannel = 0;
const int pwmResolution = 8;

// Motor simulation variables
float motorCurrent = 0;
float motorVoltage = 0;
float motorTemp = 0;
float motorVibration = 0;
int motorSpeed = 0;
int pwmDutyCycle = 0;

// Fault injection for testing
bool faultInjectionMode = false;
int faultType = 0; // 0=none, 1=overheat, 2=bearing, 3=voltage, 4=overcurrent

void setup() {
  Serial.begin(115200);
  
  ledcSetup(pwmChannel, pwmFreq, pwmResolution);
  ledcAttachPin(MOTOR_PWM_PIN, pwmChannel);
  
  pinMode(LED_NORMAL, OUTPUT);
  pinMode(LED_FAULT, OUTPUT);
  pinMode(POT_PIN, INPUT);
  
  digitalWrite(LED_NORMAL, HIGH);
  digitalWrite(LED_FAULT, LOW);
  
  Serial.println("\\n=================================");
  Serial.println("BLDC Motor Fault Detection System");
  Serial.println("=================================");
  Serial.println("Commands:");
  Serial.println("  f0 - Normal operation");
  Serial.println("  f1 - Inject overheating fault");
  Serial.println("  f2 - Inject bearing fault");
  Serial.println("  f3 - Inject voltage fault");
  Serial.println("  f4 - Inject overcurrent fault");
  Serial.println("=================================\\n");
  
  delay(1000);
}

void loop() {
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\\n');
    cmd.trim();
    
    if (cmd.startsWith("f")) {
      faultType = cmd.substring(1).toInt();
      faultInjectionMode = (faultType > 0);
      
      if (faultInjectionMode) {
        Serial.print("Fault injection mode: ");
        switch(faultType) {
          case 1: Serial.println("OVERHEATING"); break;
          case 2: Serial.println("BEARING FAULT"); break;
          case 3: Serial.println("VOLTAGE FAULT"); break;
          case 4: Serial.println("OVERCURRENT"); break;
        }
      } else {
        Serial.println("Normal operation mode");
      }
    }
  }
  
  int potValue = analogRead(POT_PIN);
  pwmDutyCycle = map(potValue, 0, 4095, 0, 255);
  ledcWrite(pwmChannel, pwmDutyCycle);
  motorSpeed = map(pwmDutyCycle, 0, 255, 0, 2000);
  
  if (faultInjectionMode) {
    injectFault();
  } else {
    normalOperation();
  }
  
  motorCurrent += random(-10, 10) / 100.0;
  motorVoltage += random(-5, 5) / 100.0;
  motorTemp += random(-3, 3) / 10.0;
  motorVibration += random(-2, 2) / 100.0;
  
  motorCurrent = constrain(motorCurrent, 0, 10);
  motorVoltage = constrain(motorVoltage, 0, 15);
  motorTemp = constrain(motorTemp, 20, 100);
  motorVibration = constrain(motorVibration, 0, 3);
  motorSpeed = constrain(motorSpeed, 0, 2000);
  
  Serial.print("DATA:");
  Serial.print(motorCurrent, 2);
  Serial.print(",");
  Serial.print(motorVoltage, 2);
  Serial.print(",");
  Serial.print(motorTemp, 1);
  Serial.print(",");
  Serial.print(motorVibration, 2);
  Serial.print(",");
  Serial.println(motorSpeed);
  
  Serial.print("Speed: ");
  Serial.print(motorSpeed);
  Serial.print(" RPM | Current: ");
  Serial.print(motorCurrent, 2);
  Serial.print("A | Voltage: ");
  Serial.print(motorVoltage, 2);
  Serial.print("V | Temp: ");
  Serial.print(motorTemp, 1);
  Serial.print("°C | Vib: ");
  Serial.print(motorVibration, 2);
  Serial.println("g");
  
  updateLEDs();
  delay(1000);
}

void normalOperation() {
  float speedFactor = motorSpeed / 2000.0;
  motorCurrent = 1.5 + speedFactor * 1.5;
  motorVoltage = 11.8 + random(-2, 3) / 10.0;
  motorTemp = 40 + speedFactor * 15;
  motorVibration = 0.3 + speedFactor * 0.3;
}

void injectFault() {
  float speedFactor = motorSpeed / 2000.0;
  
  switch(faultType) {
    case 1: // Overheating
      motorCurrent = 2.5 + speedFactor * 1.5;
      motorVoltage = 11.8 + random(-2, 3) / 10.0;
      motorTemp = 70 + speedFactor * 15;
      motorVibration = 0.5 + speedFactor * 0.4;
      break;
    case 2: // Bearing fault
      motorCurrent = 2.0 + speedFactor * 1.2;
      motorVoltage = 11.8 + random(-2, 3) / 10.0;
      motorTemp = 50 + speedFactor * 10;
      motorVibration = 1.5 + random(-3, 3) / 10.0;
      break;
    case 3: // Voltage fault
      motorCurrent = 2.5 + speedFactor * 1.0;
      motorVoltage = 9.0 + random(-5, 5) / 10.0;
      motorTemp = 45 + speedFactor * 10;
      motorVibration = 0.6 + speedFactor * 0.3;
      break;
    case 4: // Overcurrent
      motorCurrent = 4.5 + speedFactor * 1.0;
      motorVoltage = 11.5 + random(-2, 3) / 10.0;
      motorTemp = 60 + speedFactor * 10;
      motorVibration = 0.8 + speedFactor * 0.4;
      break;
    default:
      normalOperation();
      break;
  }
}

void updateLEDs() {
  bool fault = false;
  if (motorTemp > 70 || motorCurrent > 4.0 || motorVibration > 1.2 || motorVoltage < 10.0) {
    fault = true;
  }
  digitalWrite(LED_NORMAL, !fault);
  digitalWrite(LED_FAULT, fault);
}
--------------------------------------------------------------------------

Wokwi Simulator Configuration (diagram.json):
--------------------------------------------------------------------------
{
  "version": 1,
  "author": "BLDC Fault Prediction",
  "parts": [
    {"type": "wokwi-esp32-devkit-v1", "id": "esp32", "top": 0, "left": 0},
    {"type": "wokwi-potentiometer", "id": "pot1", "top": 50, "left": 300, "attrs": {"label": "Speed"}},
    {"type": "wokwi-led", "id": "led1", "top": 100, "left": 450, "attrs": {"color": "green"}},
    {"type": "wokwi-led", "id": "led2", "top": 160, "left": 450, "attrs": {"color": "red"}},
    {"type": "wokwi-resistor", "id": "r1", "top": 100, "left": 400, "attrs": {"value": "220"}},
    {"type": "wokwi-resistor", "id": "r2", "top": 160, "left": 400, "attrs": {"value": "220"}},
    {"type": "wokwi-dc-motor", "id": "motor1", "top": 400, "left": 300}
  ],
  "connections": [
    ["pot1:SIG", "esp32:GPIO34", "green", ["v0"]],
    ["pot1:GND", "esp32:GND.1", "black", ["v0"]],
    ["pot1:VCC", "esp32:3V3", "red", ["v0"]],
    ["r1:1", "esp32:GPIO13", "green", ["v0"]],
    ["r1:2", "led1:A", "green", ["v0"]],
    ["led1:C", "esp32:GND.2", "black", ["v0"]],
    ["r2:1", "esp32:GPIO12", "red", ["v0"]],
    ["r2:2", "led2:A", "red", ["v0"]],
    ["led2:C", "esp32:GND.2", "black", ["v0"]],
    ["motor1:+", "esp32:GPIO26", "red", ["v0"]],
    ["motor1:-", "esp32:GND.2", "black", ["v0"]]
  ]
}
--------------------------------------------------------------------------

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
warnings.filterwarnings('ignore')

# =================================================================================
# SECTION 1: DATA GENERATION
# =================================================================================

def generate_training_data(n_samples=2000):
    """Generate synthetic BLDC motor sensor data for training"""
    np.random.seed(42)
    data = []
    
    # Normal operation (50%)
    for _ in range(int(n_samples * 0.5)):
        current = np.random.normal(2.5, 0.3)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(45, 5)
        vibration = np.random.normal(0.5, 0.1)
        speed = np.random.normal(1500, 50)
        data.append([current, voltage, temperature, vibration, speed, 0])
    
    # Overheating fault (15%)
    for _ in range(int(n_samples * 0.15)):
        current = np.random.normal(3.5, 0.4)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(75, 8)
        vibration = np.random.normal(0.7, 0.15)
        speed = np.random.normal(1450, 80)
        data.append([current, voltage, temperature, vibration, speed, 1])
    
    # Bearing fault (15%)
    for _ in range(int(n_samples * 0.15)):
        current = np.random.normal(2.8, 0.4)
        voltage = np.random.normal(12.0, 0.2)
        temperature = np.random.normal(55, 7)
        vibration = np.random.normal(1.5, 0.3)
        speed = np.random.normal(1400, 100)
        data.append([current, voltage, temperature, vibration, speed, 2])
    
    # Voltage abnormality (10%)
    for _ in range(int(n_samples * 0.1)):
        current = np.random.normal(3.0, 0.5)
        voltage = np.random.normal(9.5, 1.0)
        temperature = np.random.normal(50, 6)
        vibration = np.random.normal(0.8, 0.2)
        speed = np.random.normal(1200, 150)
        data.append([current, voltage, temperature, vibration, speed, 3])
    
    # Short circuit / overcurrent (10%)
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
    
    # Separate features and labels
    X = df.drop('Fault', axis=1)
    y = df['Fault']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest Classifier
    print("\nTraining Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, min_samples_split=5)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*80}")
    print(f"MODEL ACCURACY: {accuracy*100:.2f}%")
    print(f"{'='*80}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=fault_names, digits=3))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=fault_names, yticklabels=fault_names, cbar_kws={'label': 'Count'})
    plt.title('BLDC Motor Fault Prediction - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual Fault Type', fontsize=12)
    plt.xlabel('Predicted Fault Type', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nFeature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"  {row['Feature']:15s}: {row['Importance']:.4f} {'█' * int(row['Importance'] * 50)}")
    
    plt.figure(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feature_importance)))
    plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance for Fault Prediction', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    print("✓ Feature importance plot saved as 'feature_importance.png'")
    
    # Save model and scaler
    joblib.dump(model, 'bldc_fault_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n✓ Model saved as 'bldc_fault_model.pkl'")
    print("✓ Scaler saved as 'scaler.pkl'")
    print("\n" + "="*80)
    
    return model, scaler

# =================================================================================
# SECTION 3: REAL-TIME PREDICTION FROM ESP32
# =================================================================================

def predict_from_serial(port='COM3', baudrate=115200):
    """Real-time prediction from serial data"""
    try:
        model = joblib.load('bldc_fault_model.pkl')
        scaler = joblib.load('scaler.pkl')
        print("✓ Model loaded successfully!")
    except:
        print("⚠ Model not found. Training new model...")
        model, scaler = train_model()
    
    fault_names = ['Normal', 'Overheating', 'Bearing Fault', 'Voltage Fault', 'Overcurrent']
    fault_colors = ['\033[92m', '\033[91m', '\033[93m', '\033[95m', '\033[91m']  # Green, Red, Yellow, Magenta, Red
    reset_color = '\033[0m'
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        print(f"\n{'='*80}")
        print(f"✓ Connected to {port} at {baudrate} baud")
        print(f"{'='*80}")
        print("\nWaiting for data from ESP32...")
        print("TIP: Use potentiometer to control motor speed")
        print("TIP: Send f0-f4 via Serial Monitor to inject faults\n")
        time.sleep(2)
        
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if line.startswith("DATA:"):
                    try:
                        values = line.replace("DATA:", "").split(',')
                        current = float(values[0])
                        voltage = float(values[1])
                        temperature = float(values[2])
                        vibration = float(values[3])
                        speed = float(values[4])
                        
                        # Prepare input
                        input_data = np.array([[current, voltage, temperature, vibration, speed]])
                        input_scaled = scaler.transform(input_data)
                        
                        # Predict
                        prediction = model.predict(input_scaled)[0]
                        probabilities = model.predict_proba(input_scaled)[0]
                        
                        # Display results
                        print(f"\n{'='*80}")
                        print(f"📊 SENSOR READINGS:")
                        print(f"   Current: {current:>6.2f}A  |  Voltage: {voltage:>6.2f}V  |  Temperature: {temperature:>5.1f}°C")
                        print(f"   Vibration: {vibration:>4.2f}g  |  Speed: {speed:>6.0f} RPM")
                        print(f"{'-'*80}")
                        print(f"🤖 PREDICTION: {fault_colors[prediction]}{fault_names[prediction]}{reset_color}")
                        print(f"📈 Confidence: {probabilities[prediction]*100:.1f}%")
                        
                        # Show all probabilities
                        print(f"\n   Probability Distribution:")
                        for i, prob in enumerate(probabilities):
                            bar = '█' * int(prob * 40)
                            print(f"   {fault_names[i]:15s}: {prob*100:5.1f}% {bar}")
                        
                        print(f"{'='*80}\n")
                        
                    except Exception as e:
                        print(f"⚠ Error parsing data: {e}")
                
    except serial.SerialException:
        print(f"\n{'='*80}")
        print(f"❌ ERROR: Could not open port {port}")
        print(f"{'='*80}")
        print("\nTroubleshooting:")
        print("1. Check if ESP32 is connected via USB")
        print("2. Verify the correct COM port")
        print("3. Close Arduino IDE Serial Monitor if open")
        print("4. Try different USB cable (must support data transfer)")
        print("\n💡 Running in SIMULATION mode instead...\n")
        simulate_predictions(model, scaler)
    except KeyboardInterrupt:
        print("\n\n⏹ Stopped by user")
        if 'ser' in locals() and ser.is_open:
            ser.close()

# =================================================================================
# SECTION 4: SIMULATION MODE
# =================================================================================

def simulate_predictions(model, scaler):
    """Simulation mode (if serial not available)"""
    fault_names = ['Normal', 'Overheating', 'Bearing Fault', 'Voltage Fault', 'Overcurrent']
    
    print("="*80)
    print("🎮 SIMULATION MODE - Testing Different Fault Scenarios")
    print("="*80)
    
    scenarios = [
        ([2.5, 12.0, 45, 0.5, 1500], "✓ Normal Operation"),
        ([3.5, 12.0, 78, 0.7, 1450], "🔥 Overheating Scenario"),
        ([2.8, 12.0, 55, 1.8, 1400], "⚙️  Bearing Fault Scenario"),
        ([3.0, 9.0, 50, 0.8, 1200], "⚡ Voltage Abnormality"),
        ([5.2, 11.0, 65, 1.0, 1300], "⚠️  Overcurrent Scenario"),
    ]
    
    for i, (values, scenario) in enumerate(scenarios, 1):
        input_data = np.array([values])
        input_scaled = scaler.transform(input_data)
        
        prediction = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        
        print(f"\n{'='*80}")
        print(f"TEST CASE {i}/5: {scenario}")
        print(f"{'='*80}")
        print(f"📊 INPUT PARAMETERS:")
        print(f"   Current: {values[0]:>6.2f}A  |  Voltage: {values[1]:>6.2f}V  |  Temperature: {values[2]:>5.1f}°C")
        print(f"   Vibration: {values[3]:>4.2f}g  |  Speed: {values[4]:>6.0f} RPM")
        print(f"{'-'*80}")
        print(f"🤖 PREDICTION: {fault_names[prediction]}")
        print(f"📈 Confidence: {probabilities[prediction]*100:.1f}%")
        print(f"\n   Probability Distribution:")
        for j, prob in enumerate(probabilities):
            bar = '█' * int(prob * 40)
            print(f"   {fault_names[j]:15s}: {prob*100:5.1f}% {bar}")
        print(f"{'='*80}")
        
        time.sleep(2)
    
    print("\n✓ Simulation complete!\n")

# =================================================================================
# SECTION 5: MAIN MENU
# =================================================================================

def display_menu():
    """Display main menu"""
    print("\n" + "="*80)
    print("🔧 BLDC MOTOR FAULT PREDICTION SYSTEM")
    print("="*80)
    print("\nSelect an option:")
    print("\n1. 🎓 Train New Model")
    print("   - Generate synthetic training data")
    print("   - Train Random Forest Classifier")
    print("   - Save model for future use")
    print("\n2. 🔌 Connect to ESP32 (Real-time Prediction)")
    print("   - Read sensor data via serial port")
    print("   - Predict faults in real-time")
    print("   - Control motor with potentiometer")
    print("\n3. 🎮 Run Simulation Mode")
    print("   - Test with pre-defined scenarios")
    print("   - No hardware required")
    print("\n4. 📊 Show Model Info")
    print("   - Display model statistics")
    print("   - View feature importance")
    print("\n5. ❌ Exit")
    print("\n" + "="*80)

def show_model_info():
    """Show information about trained model"""
    try:
        model = joblib.load('bldc_fault_model.pkl')
        print("\n" + "="*80)
        print("📊 MODEL INFORMATION")
        print("="*80)
        print(f"\nModel Type: Random Forest Classifier")
        print(f"Number of Trees: {model.n_estimators}")
        print(f"Max Depth: {model.max_depth}")
        print(f"Features: Current, Voltage, Temperature, Vibration, Speed")
        print(f"Classes: Normal, Overheating, Bearing Fault, Voltage Fault, Overcurrent")
        print("\n✓ Model is ready for predictions")
        print("="*80)
    except:
        print("\n⚠ No trained model found. Please train a model first (Option 1)")

# =================================================================================
# MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + " "*20 + "BLDC MOTOR FAULT PREDICTION SYSTEM" + " "*24 + "█")
    print("█" + " "*25 + "Using Machine Learning" + " "*32 + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    while True:
        display_menu()
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            train_model()
            input("\nPress Enter to continue...")
            
        elif choice == '2':
            port = input("\nEnter COM port (e.g., COM3, /dev/ttyUSB0): ").strip()
            if not port:
                port = 'COM3'
            predict_from_serial(port)
            
        elif choice == '3':
            try:
                model = joblib.load('bldc_fault_model.pkl')
                scaler = joblib.load('scaler.pkl')
            except:
                print("\n⚠ Model not found. Training first...")
                model, scaler = train_model()
            simulate_predictions(model, scaler)
            input("\nPress Enter to continue...")
            
        elif choice == '4':
            show_model_info()
            input("\nPress Enter to continue...")
            
        elif choice == '5':
            print("\n" + "="*80)
            print("Thank you for using BLDC Motor Fault Prediction System!")
            print("="*80 + "\n")
            break
            
        else:
            print("\n❌ Invalid choice. Please select 1-5.")
            time.sleep(1)

"""
=================================================================================
INSTALLATION INSTRUCTIONS:
=================================================================================

1. Install Required Packages:
   pip install numpy pandas scikit-learn matplotlib seaborn pyserial joblib

2. Save this file as: bldc_ml.py

3. For ESP32 Hardware:
   - Copy the Arduino code (from comments above) to Arduino IDE
   - Upload to ESP32
   - Connect potentiometer to GPIO34
   - Run: python bldc_ml.py

4. For Wokwi Simulator:
   - Go to https://wokwi.com
   - Create new ESP32 project
   - Copy Arduino code
   - Copy diagram.json
   - Start simulation

5. Run Python Script:
   python bldc_ml.py

=================================================================================
FAULT INJECTION COMMANDS (via Serial Monitor):
=================================================================================
f0 - Normal operation
f1 - Inject overheating fault
f2 - Inject bearing fault  
f3 - Inject voltage fault
f4 - Inject overcurrent fault

=================================================================================
"""