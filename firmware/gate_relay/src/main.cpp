#include <Arduino.h>

// Replaces the original (source lost) WiFi-AP relay-switch sketch recovered
// from this board's flash: same two relays (D22/D23), but controlled over
// USB serial instead of WiFi -- the university network this board would
// otherwise need to join uses enterprise auth and very likely client
// isolation, so a simple WiFi station connection wouldn't let the ALPR
// server reach it anyway. A serial tether sidesteps the network entirely.
//
// Wiring (per the physical board): D22 = gate open, D23 = gate close.
// Most small ESP32 relay modules are active-LOW (LOW energizes the relay);
// flip RELAY_ACTIVE_LOW if the relay clicks on the opposite command.
constexpr bool RELAY_ACTIVE_LOW = true;
constexpr int RELAY_OPEN_PIN = 22;
constexpr int RELAY_CLOSE_PIN = 23;
// Mimics the original UI's "press and hold" gesture (mousedown -> on,
// mouseup -> off) as a single atomic pulse -- done on-device so a serial
// hiccup on the server side can't leave a relay stuck energized.
constexpr unsigned long PULSE_MS = 400;

void setRelay(int pin, bool energize) {
  digitalWrite(pin, (energize == RELAY_ACTIVE_LOW) ? LOW : HIGH);
}

void pulseRelay(int pin, const char *name) {
  setRelay(pin, true);
  delay(PULSE_MS);
  setRelay(pin, false);
  Serial.print("OK PULSE ");
  Serial.println(name);
}

String inputBuffer;

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) {
    return;
  }
  if (cmd == "PULSE_OPEN") {
    pulseRelay(RELAY_OPEN_PIN, "OPEN");
  } else if (cmd == "PULSE_CLOSE") {
    pulseRelay(RELAY_CLOSE_PIN, "CLOSE");
  } else if (cmd == "STATUS") {
    Serial.println("OK STATUS ready");
  } else {
    Serial.print("ERR unknown command: ");
    Serial.println(cmd);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_OPEN_PIN, OUTPUT);
  pinMode(RELAY_CLOSE_PIN, OUTPUT);
  setRelay(RELAY_OPEN_PIN, false);
  setRelay(RELAY_CLOSE_PIN, false);
  Serial.println("READY gate-relay-controller");
}

void loop() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n') {
      handleCommand(inputBuffer);
      inputBuffer = "";
    } else if (c != '\r') {
      inputBuffer += c;
    }
  }
}
