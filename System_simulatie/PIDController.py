class PIDController:
    def __init__(self,kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0
    def compute(self, setpoint, current_value,dt):
        #PID berekening
        error = setpoint - current_value
        self.integral += error * dt
        derivative =( error - self.previous_error ) / dt
        self.previous_error = error
        return (self.kp * error + self.ki * self.integral +self.kd * derivative) *dt
    