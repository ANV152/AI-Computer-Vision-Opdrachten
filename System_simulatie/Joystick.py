from enum import Enum

class JoystickDirection(Enum):
    NEUTRAL = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

class Joystick:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.direction = JoystickDirection.NEUTRAL

    def set_position(self, x, y):
        self.x = x
        self.y = y
        self.update_direction()

    def isNeutral(self):
        # Neutrale drempelwaarden, bijvoorbeeld binnen ±0.1 rond de neutrale positie.
        threshold = 0.1
        return abs(self.x) < threshold and abs(self.y) < threshold

    def update_direction(self):
        if self.isNeutral():
            self.direction = JoystickDirection.NEUTRAL
        elif abs(self.y) > abs(self.x):  # Verticale beweging heeft prioriteit
            self.direction = JoystickDirection.UP if self.y > 0 else JoystickDirection.DOWN
        else:
            self.direction = JoystickDirection.RIGHT if self.x > 0 else JoystickDirection.LEFT

    def get_direction(self):
        return self.direction


