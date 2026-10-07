cmake_minimum_required(VERSION 3.20)
project(ML_From_Scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

add_library(scratch_ml STATIC
        src/supervised/ID3.cpp
        include/scratch_ml/supervised/Classifier.hpp
        include/scratch_ml/core/matrix.hpp
        include/scratch_ml/supervised/ID3.hpp
        include/scratch_ml/supervised/CART.hpp
        src/supervised/CART.cpp)

target_include_directories(scratch_ml PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(ml_runner tests/main.cpp
        src/core/matrix.cpp
)

target_link_libraries(ml_runner PRIVATE scratch_ml)