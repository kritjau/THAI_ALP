# Gate relay firmware (ESP32-WROOM-32)

Replaces the board's original firmware (source lost -- recovered only by
dumping and reading its compiled flash). That version ran its own isolated
WiFi access point with an HTTP relay-control API; this one drops WiFi
entirely and takes commands over USB serial instead, since the server
controlling the gate is on a university network with enterprise auth and
likely client isolation -- a WiFi station connection wouldn't have been
reachable from the server anyway. A serial tether sidesteps the network
question altogether.

## Wiring

- Relay 1 -> GPIO 22 -> gate **open**
- Relay 2 -> GPIO 23 -> gate **close**
- Most small ESP32 relay modules are active-LOW (LOW energizes the relay).
  `RELAY_ACTIVE_LOW` in `src/main.cpp` assumes this -- flip it if a relay
  clicks on the opposite command from what's sent.

## Serial protocol

115200 baud. One command per line (`\n`-terminated):

- `PULSE_OPEN` -- energizes the open relay for 400ms then releases it
  (mimics a physical button press-and-release), replies `OK PULSE OPEN`
- `PULSE_CLOSE` -- same, for the close relay, replies `OK PULSE CLOSE`
- `STATUS` -- replies `OK STATUS ready` (no relay movement -- just confirms
  the board is alive and listening)
- Anything else replies `ERR unknown command: <text>`

`app/gate.py` is the only code that speaks this protocol.

## Building and flashing

```bash
python3 -m venv .venv-fw && source .venv-fw/bin/activate   # keep this
                                                             # separate from
                                                             # the app's own
                                                             # venv
pip install platformio
cd firmware/gate_relay
platformio run                    # compile
platformio run --target upload    # flash over USB (board must be at
                                   # whatever serial port platformio.ini's
                                   # upload_port points to -- adjust if the
                                   # server enumerates it differently, e.g.
                                   # not /dev/ttyUSB0)
```

## Manual test after flashing

With nothing else talking to the port:

```python
import serial, time
ser = serial.Serial('/dev/ttyUSB0', 115200, timeout=2)
time.sleep(2)
ser.write(b'STATUS\n'); print(ser.readline())      # safe, no relay movement
# Only once the gate area is confirmed clear:
ser.write(b'PULSE_OPEN\n'); print(ser.readline())  # actually moves the gate
```
