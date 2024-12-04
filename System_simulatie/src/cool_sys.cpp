#include <pybind11/pybind11.h>

#include "motor.hpp"
// #include "fan.cpp"
// #include "steppermotor.cpp"
#include "thermometer.cpp"
namespace py = pybind11;

PYBIND11_MODULE(cool_sys, m) {
    py::class_<Thermometer>(m, "Thermometer")
        .def(py::init<>())
        .def("updateTemperature", &Thermometer::updateTemperature, 
             "Update the temperature based on motor status and fan speed",
             py::arg("motor_running"), py::arg("fan_speed"))
        .def("getTemperature", &Thermometer::getTemperature, 
             "Get the current temperature reading");
    py::class_<Motor>(m, "Motor")
    
        .def("setSpeed", &Motor::setSpeed)
        .def("getSpeed", &Motor::getSpeed)
        .def("stop", &Motor::stop);
    py::class_<Fan,Motor>(m, "Fan")
        .def(py::init<>())
        .def("setSpeed", &Fan::setSpeed, "Set the speed of the fan")
        .def("getSpeed", &Fan::getSpeed, "geef de snelheid van de fan terug");

    py::class_<StepperMotor, Motor>(m, "StepperMotor")
        .def(py::init<>())
        .def("setSpeed", &StepperMotor::setSpeed, "zet de snelheid van de motor en schakkelt naar microstepping als de snelheid heel lag is")
        .def("getSpeed", &StepperMotor::getSpeed, "geeft de snelheid terug")
        .def("stop", &StepperMotor::stop, "Stop de motor");
}
