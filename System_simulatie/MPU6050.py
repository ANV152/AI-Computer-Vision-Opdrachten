import random 
from Sensor import Sensor
from Decorators import *
import time
class MPU6050(Sensor):
    def __init__(self, object_id):
        self.angle = 1
        self.object_id = object_id
        self.angular_velocity = 0.0
        self.time_constant = 0.01
    def get_angle(self):
        return self.angle
    
    def update(self, motor_input, time_step):
        # Simuleer de verandering in hoek op basis van motorinput en tijdsinterval
        # new_angle = old_angle + angular_velocity * time_step
        self.angular_velocity += motor_input * time_step  # Hoeksnelheid op basis van motorinvoer
        self.angle += self.angular_velocity * time_step  # Hoek bijwerken
    # @timeout(5.0)
    # def calibrate(self):
    #     time.sleep(6)


