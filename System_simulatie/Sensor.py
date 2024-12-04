import abc 
class Sensor:
    __metaclass__ = abc.ABCMeta
    def __init__(self):
        self.value = None
    @abc.abstractmethod
    def start(self):
        pass
    @abc.abstractmethod
    def readValue(self):
        return self.value
    @abc.abstractmethod
    def calibrate(self):
        pass