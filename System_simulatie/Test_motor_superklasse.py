import unittest
 # Importeer jouw Joystick class
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cool_sys

class MotorControlTest(unittest.TestCase):
    def test_motor_control(self):
        # Creëer instanties van Fan en StepperMotor
        fan = cool_sys.Fan()
        stepper_motor = cool_sys.StepperMotor()

        # Test de fan-motor aansturing
        fan.set_speed(50)  # Bijvoorbeeld een snelheid van 50 voor de ventilator
        self.assertEqual(fan.speed, 50)

        # Test de stepper motor aansturing
        stepper_motor.set_speed(100)  # Bijvoorbeeld een snelheid van 100 voor de steppermotor
        self.assertEqual(stepper_motor.speed, 100)

        # Check dat het juiste type object wordt aangestuurd
        self.assertIsInstance(fan, cool_sys.Motor())
        self.assertIsInstance(stepper_motor, cool_sys.Motor())

if __name__ == "__main__":
    unittest.main()
