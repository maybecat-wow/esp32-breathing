# Top-level ESP-IDF project for the Mecha smart-plug firmware.
# Build from this directory after `source ~/.espressif/tools/activate_idf_v6.0.sh`.
cmake_minimum_required(VERSION 3.5)
add_compile_options(-fdiagnostics-color=always)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(mecha_smart_plug)
