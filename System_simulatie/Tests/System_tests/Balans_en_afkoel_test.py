import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import unittest
import time
from Robot import Robot
from Joystick import Joystick
class SystemTestBalancingAndCooling(unittest.TestCase):
    def setUp(self):
        # Initialiseer de robot, joystick, sensor en ventilator
        self.robot = Robot()
        self.joystick = Joystick()
        self.robot.joystick = self.joystick  # Verbind joystick met de robot

    def test_balancing_and_cooling(self):
        # Stap 1: Test normale omstandigheden (lichte bewegingen)
        self.joystick.set_position(0, 1)  # Joystick omhoog
        for _ in range(100):  # Simuleer 100 tijdstappen
            self.robot.update()
            self.assertTrue(abs(self.robot.angle) < 10, "Robot is niet in balans!")
            time.sleep(0.01)  # Wacht even tussen elke update voor de simulatie

        # Stap 2: Test zware omstandigheden (continue bewegingen)
        self.joystick.set_position(0, 1)  # Joystick omhoog
        for _ in range(300):  # Simuleer langere belasting
            self.robot.update()
            # Verhoog de motor temperatuur om zware belasting te simuleren
            self.robot.term_sensor.increase_temperature(0.05)  # Verhoog temperatuur per tijdstap
            time.sleep(0.01)

        # Stap 3: Check of de ventilator automatisch aan gaat bij hogere temperatuur
        self.assertTrue(self.robot.fan.getSpeed() > 0, "Ventilator ging niet aan bij hoge temperatuur")
        
        # Stap 4: Controleer dat de ventilator weer uit gaat als de temperatuur daalt
        while self.robot.term_sensor.get_temperature() > 20:
            self.robot.term_sensor.decrease_temperature(0.1)  # Simuleer afkoelen
            self.robot.update()
            time.sleep(0.01)

        self.assertTrue(self.robot.fan.getSpeed() == 0, "Ventilator bleef aan bij normale temperatuur")

if __name__ == "__main__":
    unittest.main()
