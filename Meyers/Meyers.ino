#include <Servo.h>
#define pinServo A0

Servo myservo;  // create servo object to control a servo

int pos = 0;    // variable to store the servo position

void setup() 
{
  myservo.attach(pinServo); //analog pin 0
  Serial.begin(115200); // Python script will talk to arduino serially
}

void loop() 
{
  // Pretty simple here. Just sit on the serial port and 
  // wait for ints to come in
  while(Serial.available()) 
  {
    pos = Serial.parseInt(); // get the angle from the python script
    if(pos > 0) myservo.write(pos); // turn the servo
    delay(15); // slight delay
  }
}

