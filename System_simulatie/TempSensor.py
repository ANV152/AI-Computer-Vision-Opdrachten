from Sensor import Sensor
from Decorators import *
class TemperatureSensor(Sensor):
    def __init__(self, object_id):
        self.object_id = object_id
    def get_data(self):
        # Simulatie van het ophalen van temperatuurgegevens
        return 
    @timeout(5.0)
    def calibrate(self):
        time.sleep(4)
        return 