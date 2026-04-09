cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
project(footballnet LANGUAGES CXX)

# =========================
# Compiler settings
# =========================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# OpenCV
# =========================
find_package(OpenCV REQUIRED)
message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")

# =========================
# JsonCpp
# =========================
find_package(jsoncpp REQUIRED)

# =========================
# LibTorch
# =========================
set(Torch_DIR ${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch)    
find_package(Torch REQUIRED)
message(STATUS "Found Torch")

# =========================
# Executable
# =========================
add_executable(footballnet
    inference.cpp
)

# =========================
# Include paths
# =========================
target_include_directories(footballnet PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

# =========================
# Link libraries
# =========================
target_link_libraries(footballnet
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    jsoncpp
)

# =========================
# Torch compile flags
# =========================
set_property(TARGET footballnet PROPERTY CXX_STANDARD 17)

# Fix for Raspberry Pi + GCC
target_compile_definitions(footballnet PRIVATE
    -D_GLIBCXX_USE_CXX11_ABI=1
)

# =========================
# RPATH (important on Pi)
# =========================
set_target_properties(footballnet PROPERTIES
BUILD_RPATH "$ORIGIN"
INSTALL_RPATH "$ORIGIN"
)