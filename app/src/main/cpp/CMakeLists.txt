cmake_minimum_required(VERSION 3.22.1)

project("voiceassistant")

# Set C++ standard to C++17 for filesystem support
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for ONNX Runtime
include_directories(${CMAKE_SOURCE_DIR}/include)

# Find ONNX Runtime prebuilt library
set(ONNX_RUNTIME_LIB_DIR ${CMAKE_SOURCE_DIR}/libs/${ANDROID_ABI})
find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNX_RUNTIME_LIB_DIR} NO_CMAKE_FIND_ROOT_PATH REQUIRED)

# Add wakeupdetecter library
add_library(${CMAKE_PROJECT_NAME} SHARED
        native-lib.cpp
        wakeup_detector.cpp)

# Find required libraries
find_library(log-lib log)

# Link libraries to our target
target_link_libraries(${CMAKE_PROJECT_NAME}
        ${ONNXRUNTIME_LIB}
        android
        ${log-lib})