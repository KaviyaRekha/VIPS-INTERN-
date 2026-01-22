// Simulation-based BLDC Motor Fault Prediction (ESP32 + Tiny ML)
// Wokwi: DC motor is used as BLDC emulator

#define POT_PIN 34
#define MOTOR_PWM_PIN 26
#define LED_NORMAL 13
#define LED_FAULT 12

const int pwmFreq = 5000;
const int pwmResolution = 8; // 0..255 duty

// Features (simulated sensors)
float I = 0.0;   // current (A)
float V = 0.0;   // voltage (V)
float T = 0.0;   // temp (C)
float Vib = 0.0; // vibration (g)

int speedRPM = 0;
int duty = 0;

// Manual forcing faults from serial (f0..f4)
bool forceFault = false;
int forcedType = 0; // 0 none, 1 overheat, 2 bearing, 3 voltage, 4 overcurrent/short

float clampf(float x, float lo, float hi) {
  if (x < lo) return lo;
  if (x > hi) return hi;
  return x;
}

float noise(float amp) {
  return (random(-100, 101) / 100.0) * amp;
}

// ----------- Emulated motor physics / sensor abstraction -----------
void simulateNormal(float sf) {
  I   = 1.2 + 1.8 * sf;           // 1.2..3.0 A
  V   = 12.0 + noise(0.25);       // ~12V
  T   = 35.0 + 22.0 * sf;         // 35..57 C
  Vib = 0.25 + 0.35 * sf;         // 0.25..0.60 g
}

void simulateFault(float sf, int type) {
  switch (type) {
    case 1: // OVERHEAT
      I   = 2.3 + 1.6 * sf;
      V   = 12.0 + noise(0.25);
      T   = 75.0 + 18.0 * sf;
      Vib = 0.55 + 0.40 * sf;
      break;

    case 2: // BEARING fault
      I   = 2.0 + 1.4 * sf;
      V   = 12.0 + noise(0.25);
      T   = 45.0 + 12.0 * sf;
      Vib = 1.4 + noise(0.25);
      break;

    case 3: // VOLTAGE abnormal
      I   = 2.3 + 1.2 * sf;
      V   = 9.2 + noise(0.45);
      T   = 42.0 + 12.0 * sf;
      Vib = 0.55 + 0.30 * sf;
      break;

    case 4: // OVERCURRENT / SHORT
      I   = 4.8 + 1.2 * sf;
      V   = 11.5 + noise(0.25);
      T   = 60.0 + 12.0 * sf;
      Vib = 0.85 + 0.45 * sf;
      break;

    default:
      simulateNormal(sf);
      break;
  }
}

// ----------- Tiny ML model (on-device classifier) -----------
enum FaultClass {
  NORMAL = 0,
  OVERHEAT = 1,
  BEARING = 2,
  VOLTAGE_ABNORMAL = 3,
  OVERCURRENT_SHORT = 4
};

const char* className(int c) {
  switch (c) {
    case NORMAL: return "NORMAL";
    case OVERHEAT: return "OVERHEAT";
    case BEARING: return "BEARING";
    case VOLTAGE_ABNORMAL: return "VOLTAGE_ABNORMAL";
    case OVERCURRENT_SHORT: return "OVERCURRENT_SHORT";
    default: return "UNKNOWN";
  }
}

float riskAbove(float x, float thr, float span) {
  return clampf((x - thr) / span, 0.0, 1.0);
}
float riskBelow(float x, float thr, float span) {
  return clampf((thr - x) / span, 0.0, 1.0);
}

void predictFault(int &predClass, float &conf) {
  float rT  = riskAbove(T, 70.0, 15.0);
  float rV  = riskBelow(V, 10.0, 2.0);
  float rI  = riskAbove(I, 4.0, 2.0);
  float rVB = riskAbove(Vib, 1.2, 0.8);

  float sNormal   = 1.0 - clampf((rT + rV + rI + rVB) / 2.2, 0.0, 1.0);
  float sOverheat = 0.65 * rT + 0.15 * riskAbove(I, 3.0, 2.0) + 0.10 * riskAbove(T, 80.0, 10.0);
  float sBearing  = 0.75 * rVB + 0.15 * riskAbove(I, 2.5, 2.0) + 0.05 * riskAbove(T, 50.0, 20.0);
  float sVoltage  = 0.80 * rV + 0.10 * riskAbove(I, 2.5, 2.0);
  float sOverCur  = 0.85 * rI + 0.10 * riskAbove(T, 55.0, 25.0);

  float scores[5] = { sNormal, sOverheat, sBearing, sVoltage, sOverCur };

  int best = 0;
  float bestVal = scores[0];
  float sum = scores[0];

  for (int k = 1; k < 5; k++) {
    sum += scores[k];
    if (scores[k] > bestVal) {
      bestVal = scores[k];
      best = k;
    }
  }

  conf = (sum > 0.0001) ? (bestVal / sum) : 1.0;
  predClass = best;
}

// ----------- Serial commands -----------
void handleSerial() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();
  cmd.toLowerCase();

  if (cmd.startsWith("f")) {
    int x = cmd.substring(1).toInt();
    if (x < 0) x = 0;
    if (x > 4) x = 4;

    forcedType = x;
    forceFault = (forcedType != 0);

    if (!forceFault) Serial.println("Mode: AUTO faults by potentiometer zones");
    else {
      Serial.print("Mode: FORCED fault f"); Serial.println(forcedType);
    }
  }
}

// ----------- NEW: Auto fault zones using ONLY potentiometer speed -----------
int autoFaultFromSpeed(float sf) {
  // sf 0..1
  // 0.00 - 0.55  -> NORMAL
  // 0.55 - 0.70  -> VOLTAGE abnormal (simulate supply drop mid range)
  // 0.70 - 0.85  -> BEARING (vibration increases)
  // 0.85 - 0.95  -> OVERHEAT
  // 0.95 - 1.00  -> OVERCURRENT/SHORT
  if (sf < 0.55) return 0;
  if (sf < 0.70) return 3;
  if (sf < 0.85) return 2;
  if (sf < 0.95) return 1;
  return 4;
}

void setup() {
  Serial.begin(115200);
  delay(300);

  ledcAttach(MOTOR_PWM_PIN, pwmFreq, pwmResolution);

  pinMode(POT_PIN, INPUT);
  pinMode(LED_NORMAL, OUTPUT);
  pinMode(LED_FAULT, OUTPUT);

  digitalWrite(LED_NORMAL, HIGH);
  digitalWrite(LED_FAULT, LOW);

  randomSeed(analogRead(POT_PIN));

  Serial.println("\n=== BLDC Fault Prediction (SIM + Tiny ML) ===");
  Serial.println("Pot controls speed (PWM -> DC motor as BLDC emulator).");
  Serial.println("AUTO faults shown using pot zones OR type f1..f4 to force.");
  Serial.println("Commands: f0=auto, f1=overheat, f2=bearing, f3=voltage, f4=overcurrent");
  Serial.println("============================================\n");
}

void loop() {
  handleSerial();

  // Pot -> PWM duty
  int pot = analogRead(POT_PIN);              // 0..4095
  duty = map(pot, 0, 4095, 0, 255);           // 0..255
  ledcWrite(MOTOR_PWM_PIN, duty);             // PWM to motor

  // Duty -> speed
  speedRPM = map(duty, 0, 255, 0, 2000);
  float sf = speedRPM / 2000.0;               // 0..1

  // Simulate features: forced fault OR auto fault zones
  if (forceFault) {
    simulateFault(sf, forcedType);
  } else {
    int autoType = autoFaultFromSpeed(sf);
    if (autoType == 0) simulateNormal(sf);
    else simulateFault(sf, autoType);
  }

  // Noise + clamp
  I   = clampf(I + noise(0.08), 0.0, 10.0);
  V   = clampf(V + noise(0.05), 0.0, 15.0);
  T   = clampf(T + noise(0.30), 20.0, 100.0);
  Vib = clampf(Vib + noise(0.03), 0.0, 3.0);

  // ML prediction
  int pred;
  float conf;
  predictFault(pred, conf);

  // LEDs (based on ML prediction)
  bool fault = (pred != NORMAL);
  digitalWrite(LED_NORMAL, fault ? LOW : HIGH);
  digitalWrite(LED_FAULT,  fault ? HIGH : LOW);

  // Output
  Serial.print("PWM=");
  Serial.print(duty);
  Serial.print(" | FEATURES:");
  Serial.print(I, 2); Serial.print(",");
  Serial.print(V, 2); Serial.print(",");
  Serial.print(T, 1); Serial.print(",");
  Serial.print(Vib, 2); Serial.print(",");
  Serial.print(speedRPM);

  Serial.print(" | PRED=");
  Serial.print(className(pred));
  Serial.print(" conf=");
  Serial.println(conf, 2);

  delay(1000);
}
