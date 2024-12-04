#ifndef MOTOR_HPP
#define MOTOR_HPP

#include <iostream>
#include <iostream>
#include <chrono>
#include <thread>
#include "motor.hpp"

class Motor {
protected:
    int speed;
public:
    Motor() : speed(0) {}  
    virtual ~Motor() = default;
    virtual void setSpeed(float new_speed) = 0;
    virtual float getSpeed() const = 0;
    virtual void stop() = 0;
};

class StepperMotor : public Motor {
    private:
        void step(int direction) {
            currentPosition += direction;
        }

        int stepsPerRev;
        double stepAngle;
        int microstepping;
        int currentPosition;
        double holdTorque;
        float speed;
        std::chrono::steady_clock::time_point lastStepTime; // Last step tijdstip
public:
    StepperMotor() : Motor(), currentPosition(0), stepAngle(1.8), microstepping(16), holdTorque(4.08), speed(0) {
        // Bereken stappen per revolutie op basis van stapgrootte en microstepping
        stepsPerRev = static_cast<int>(360.0 / (stepAngle / microstepping));
        lastStepTime = std::chrono::steady_clock::now(); //initialiseert de tijd van de laatste stap
    }

    void setSpeed(float new_speed) override {
        speed = new_speed;  // Stel snelheid in
        std::cout << "StepperMotor snelheid ingesteld op " << speed << " stappen per seconde." << std::endl;
    }

    float getSpeed() const override {
        return speed;
    }

    void stop() override {
        speed = 0;
        std::cout << "StepperMotor gestopt." << std::endl;
    }

    void applyPIDOutput(double pidOutput) {
        int steps = static_cast<int>(pidOutput);
        moveToPosition(currentPosition + steps);
    }

    void moveToPosition(int targetPos) {
        int stepsNeeded = targetPos - currentPosition;
        int direction = (stepsNeeded > 0) ? 1 : -1;

        while (currentPosition != targetPos) {
            auto now = std::chrono::steady_clock::now();
            auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(now - lastStepTime);

            if (elapsedTime.count() >= (1000 / std::abs(speed))) {
                step(direction);
                lastStepTime = now;  // Werk `lastStepTime` bij
            }
        }
    }

    int getCurrentPosition() const {
        return currentPosition;
    }


};
class Fan : public Motor{
public:
    Fan() : Motor(){}

    void setSpeed(float new_speed) override {
        speed = new_speed;
        std::cout << "Ventilator snelheid ingesteld op " << speed << " RPM." << std::endl;
    }

    float getSpeed() const override {
        return speed;
    }

    void stop() override {
        speed = 0;
        std::cout << "Ventilator gestopt." << std::endl;
    }
};
#endif