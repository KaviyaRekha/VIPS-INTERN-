// BLDC Motor Fault Detection - ESP32
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
    String cmd = Serial.readStringUntil('\n');
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
}