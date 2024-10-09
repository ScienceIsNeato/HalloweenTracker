#include <Servo.h>

#define pinServo 9  // Use a PWM-capable pin (e.g., Pin 9)

Servo myservo;  // Create servo object to control a servo

int pos = 0;    // Variable to store the servo position
bool handshake_done = false;  // Flag to indicate handshake completion

void setup() 
{
  myservo.attach(pinServo); // Attach servo to Pin 9
  Serial.begin(9600);     // Initialize serial communication at 9600 baud
  Serial.println("Arduino Ready");

  // Handshake with Python
  while (!handshake_done)
  {
    if (Serial.available())
    {
      String message = Serial.readStringUntil('\n');
      message.trim();
      if (message == "PYTHON_HANDSHAKE")
      {
        Serial.println("ARDUINO_HANDSHAKE");
        handshake_done = true;
      }
    }
  }

  Serial.println("Handshake complete. Ready to receive positions.");
}

void loop() 
{
  if (handshake_done && Serial.available())
  {
    String input = Serial.readStringUntil('\n');
    input.trim();
    pos = input.toInt(); // Get the angle from the Python script

    if (pos >= 0 && pos <= 180)
    {
      myservo.write(pos); // Turn the servo
      Serial.print("Moved to position: ");
      Serial.println(pos);
    }
    else
    {
      Serial.print("Invalid position received: ");
      Serial.println(pos);
    }
    delay(15); // Slight delay
  }
}
