#ifndef THERMOMETER_CPP
#define THERMOMETER_CPP
class Thermometer {
private:
    double temperature;

public:
    Thermometer() : temperature(20.0) {}

    void updateTemperature(bool motor_running, double fan_speed) {
        if (motor_running) {
            temperature += 0.5;  // Stijgt als motor draait
        }
        if (fan_speed > 0.0) {
            temperature -= fan_speed * 0.3;  // Koelt af afhankelijk van ventilatorsnelheid
        }
        // Beperk temperatuur
        if (temperature > 100.0) temperature = 100.0;
        if (temperature < 20.0) temperature = 20.0;
    }

    double getTemperature() const {
        return temperature;
    }
};
#endif