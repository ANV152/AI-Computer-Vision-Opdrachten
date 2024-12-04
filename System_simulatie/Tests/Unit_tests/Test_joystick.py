import unittest
 # Importeer jouw Joystick class
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from Joystick import Joystick 
class TestJoystickNeutral(unittest.TestCase):
    def setUp(self):
        self.joystick = Joystick()

    def test_neutral_position(self):
        
        self.joystick.set_position(0.0, 0.0)
        self.assertTrue(self.joystick.isNeutral(), "Joystick should be neutral at (0.0, 0.0)")

        
        self.joystick.set_position(0.05, -0.05)
        self.assertTrue(self.joystick.isNeutral(), "Joystick should be neutral within threshold (0.05, -0.05)")

    def test_non_neutral_position(self):
        
        self.joystick.set_position(0.2, 0.0)
        self.assertFalse(self.joystick.isNeutral(), "Joystick should not be neutral at (0.2, 0.0)")

        self.joystick.set_position(0.0, -0.15)
        self.assertFalse(self.joystick.isNeutral(), "Joystick should not be neutral at (0.0, -0.15)")
        self.joystick.set_position(1.0, -1.0)
        self.assertFalse(self.joystick.isNeutral(), "Joystick should not be neutral at large fluctuations (1.0, -1.0)")

if __name__ == "__main__":
    unittest.main()
