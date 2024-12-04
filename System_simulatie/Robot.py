""" Python 3.10 wordt voor dit project gebruikt """
import cool_sys
from Joystick import *
# from Tempgyro import Temperaturegyro
from MPU6050 import MPU6050
from PIDController import PIDController
from Decorators import *
import matplotlib.pyplot as plt
import numpy as np
import cool_sys #Hierin bevinden zich de termometer en de ventilator
class Robot:
    def __init__(self):
        # Environment properties
        self.speed_constant = 0.1
        # Actuators
        self.left_motor = cool_sys.StepperMotor()
        self.right_motor = cool_sys.StepperMotor()
        self.fan = cool_sys.Fan()
        # PID Controllers
        self.angle_pid = PIDController(1.5, 0.5, 0.5)
        self.temp_pid = PIDController(1.0, 0.05, 0.1)
        self.speed_pid = PIDController(1.0, 0.1, 0.02)
        # Sensors
        self.term_sensor = cool_sys.Thermometer()
        self.gyro = MPU6050(0)
        self.joystick = Joystick()
        # Parameters
        self.angle = 0.0
        self.setpoint_angle = 0
        self.setpoint_temp = 20.0
        self.dt = 0.01

    def calculate_motor_input(self, target_speed, current_speed):
        # Similar logic as before for speed error
        speed_error = target_speed - current_speed
        motor_input = self.speed_pid.compute(0, speed_error, self.dt)
        max_motor_input = 100.0
        return max(-max_motor_input, min(motor_input, max_motor_input))
    
    def calculate_control_inputs(self):
        if self.joystick.get_direction() == JoystickDirection.UP:
            self.setpoint_angle += 1
        elif self.joystick.get_direction() == JoystickDirection.DOWN:
            self.setpoint_angle -= 1

        
        current_angle = self.gyro.get_angle()
        angle_motor_input = self.angle_pid.compute(self.setpoint_angle, current_angle, self.dt)

        
        target_speed = self.calculate_actual_speed(self.setpoint_angle, angle_motor_input)
        current_speed = self.calculate_actual_speed(current_angle, angle_motor_input)
        speed_motor_input = self.calculate_motor_input(target_speed, current_speed)

       
        current_temp = self.term_sensor.getTemperature()
        fan_speed = self.temp_pid.compute(self.setpoint_temp, current_temp, self.dt)

        return angle_motor_input + speed_motor_input, fan_speed
    def calculate_actual_speed(self, tilt_angle, motor_speed):
        
        speed = self.speed_constant * abs(tilt_angle) * motor_speed
        
        max_simulated_speed = 100.0  
        return min(speed, max_simulated_speed)

    def update(self):
    
        motor_input, fan_speed = self.calculate_control_inputs()

  
        self.left_motor.setSpeed(motor_input)
        self.right_motor.setSpeed(motor_input)
        self.fan.setSpeed(fan_speed)

      
        self.gyro.update(motor_input, self.dt)
        self.term_sensor.updateTemperature(self.fan.getSpeed(), self.dt)

    def stop(self):
        self.left_motor.stop()
        self.right_motor.stop()



    

################################# Simuleer de robot voor een bepaalde tijdperiode ##################
# import time

# robot = Robot()

# for step in range(1000):
#     robot.update()  # Update all components
#     time.sleep(0.01)

# robot.stop()

    
