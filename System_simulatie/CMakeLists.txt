cmake_minimum_required(VERSION 3.12)
project(System_simulatie)

add_subdirectory(pybind11)  #Gebruik de lokale pybind11-map

pybind11_add_module(cool_sys src/cool_sys.cpp)

