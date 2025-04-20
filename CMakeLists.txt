cmake_minimum_required(VERSION 3.23)

#Define project.
project(
  Neural-for-USD
  VERSION 0.0.1
  DESCRIPTION ""
  LANGUAGES CXX)

include_directories(./include)

#Discover cmake modules.
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/
                      ${CMAKE_CURRENT_SOURCE_DIR}/cmake/packages)

#Exposes build options(with default values) for this project.
include(Options)

#Set project defaults.
include(Defaults)

#Import the package dependencies(USD, TBB, ...).
include(Packages)

#Include CMake utilities for USD projects.
include(Utils)

#Recurse into source tree.
add_subdirectory(src)
