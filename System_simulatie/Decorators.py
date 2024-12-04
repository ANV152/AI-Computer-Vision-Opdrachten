import time
import functools
import signal
import threading
"""
    Next to pure functionalities of decorators:
    - Security
    - Logging
    - Serialisation
    - Dispatching
    - Deligating
    - timing

"""
# deze functie is voor het schrijven naar een bestand
def log_error(message, log_file):
    with open(log_file, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
class TimeoutException(Exception):
    def __init__(self, func_name, class_name , object_id = None):
        # super().__init__(message)
        self.func_name = func_name
        self.class_name = class_name
        self.object_id = object_id
    def __str__(self):
        return f"Timeout in functie '{self.func_name} uit klasse: {self.class_name} en object-id: {self.object_id}"
# def timeout(seconds):
# Deze decorator is alleen bedoeld voor het testen van klasse methodes en niet loze functies. 
# De reden is het gebruik van "self" die niet in een losse functie kan worden toegepast 
def timeout(seconds: float, log_file="errors.log"):
    def decorator(func):            
        @functools.wraps(func)
        def wrapper(self,*args, **kwargs):
            print("entering :", self.__class__.__name__ ,self.object_id)
            class_name = self.__class__.__name__

            result_container = []
            exception_container = []
            def uitvoeren():# de module threads heeft straks deze functie nodig
                #hieronder voeren wij de echte functie uit
                try:
                    result_container.append(func(self, *args, **kwargs))
                except Exception as e:
                    exception_container.append(e) #zoals eerder gezegd: excptions tijdens thread -> in container gegooid
            thread = threading.Thread(target=uitvoeren)#hier wordt voordeel genomen van het feite dat python functies ook als objecten ziet. 
                                                    #op deze manier gooien wij de eerdere target functie waarmee de gewrappte functie wordt uitgevoerd in de thread 
            thread.start()
            thread.join(timeout = seconds)
             # met .is_alive checken wij of de timeout
            if thread.is_alive():
                error_message = (f"Timeout van {seconds} seconden overschreden. "
                                 f"Functie: {func.__name__}, Klasse: {class_name}, "
                                 f"Object-ID: {self.object_id}\n")
                log_error(error_message, log_file)
                return None # hier retourneer ik None zodat de code verder kan draaien
            return result_container[0] if result_container else None
        return wrapper
    return decorator
# @timeout(5)
# def calibrate_sensor(sensor_id):
#     print(f"Ijking van sensor {sensor_id}...")
#     # Simuleer een langdurig proces
#     time.sleep(10)  # Dit zal de timeout overschrijden
#     print(f"Sensor {sensor_id} is gekalibreerd.")
#     return "Ijking voltooid."
