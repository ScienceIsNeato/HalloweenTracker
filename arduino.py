import serial
import threading
import time
import sys
from serial.tools import list_ports

class ArduinoController:
    def __init__(self, serial_port='COM5'):
        self.serial_port = serial_port
        self.arduino_enabled = True
        self.ser = None
        self.serial_thread = None
        self.serial_command_queue = []
        self.handshake_done = threading.Event()
        self.serial_thread_stop_event = threading.Event()
        self.initialize_serial_connection()

    def initialize_serial_connection(self):
        try:
            self.ser = serial.Serial(self.serial_port, 9600, timeout=1)
            print(f"Arduino connected on {self.serial_port}.")
            time.sleep(2)  # Wait for Arduino to reset

            # Start the serial communication thread
            self.serial_thread = threading.Thread(target=self.serial_communication_thread)
            self.serial_thread.daemon = True
            self.serial_thread.start()

            # Perform handshake
            self.perform_handshake()

        except serial.SerialException:
            self.arduino_enabled = False
            self.ser = None
            print(f"\nWARNING: Arduino not connected on Serial Port '{self.serial_port}'.")

            # List available serial ports
            available_ports = list_ports.comports()
            if available_ports:
                print("\nAvailable serial ports:")
                for port in available_ports:
                    print(f"  {port.device}")
                print("\nYou can re-run the program with the appropriate flag to choose the correct port:")
                print(f"  python {sys.argv[0]} --serial_port <port>")
            else:
                print("\nNo serial ports found.")

            input("\nPress any key to continue...")

    def serial_communication_thread(self):
        while True:
            # Send commands if any
            if self.serial_command_queue:
                command = self.serial_command_queue.pop(0)
                self.ser.write((command + '\n').encode())
                print(f"Sent to Arduino: {command}")

            # Read from serial port
            try:
                if self.ser.in_waiting > 0:
                    response = self.ser.readline().decode().strip()
                    if response:
                        print(f"Arduino: {response}")
                        if response == "ARDUINO_HANDSHAKE":
                            self.handshake_done.set()
            except Exception as e:
                print(f"Error reading from serial port: {e}")
                break  # Exit the thread on error

            time.sleep(0.01)  # Small delay to prevent high CPU usage

    def perform_handshake(self):
        print("Starting handshake with Arduino...")

        # Send handshake initiation
        self.ser.write("PYTHON_HANDSHAKE\n".encode())
        print("Sent 'PYTHON_HANDSHAKE' to Arduino.")

        # Wait for Arduino's response
        start_time = time.time()
        timeout = 5  # seconds
        while not self.handshake_done.is_set():
            if time.time() - start_time > timeout:
                print("Handshake failed: Timeout waiting for Arduino response.")
                self.arduino_enabled = False
                break
            time.sleep(0.1)
        if self.handshake_done.is_set():
            print("Handshake successful!")

    def send_servo_position(self, pos):
        if not self.arduino_enabled or self.ser is None:
            return
        if not (0 <= pos <= 180):
            print(f"Invalid position value: {pos}. Must be between 0 and 180.")
            return
        command = f"{pos}"
        self.serial_command_queue.append(command)
        print(f"Queued servo position: {pos}")

    def close(self):
        if self.ser is not None and self.ser.is_open:
            self.ser.close()
        if self.serial_thread is not None:
            self.serial_thread.join()
