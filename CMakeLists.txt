cmake_minimum_required(VERSION 3.20)
project(ML_From_Scratch LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

include_directories(include)

add_library(scratch_ml STATIC
        src/ID3.cpp)

target_include_directories(scratch_ml PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

add_executable(ml_runner src/main.cpp)

target_link_libraries(ml_runner PRIVATE scratch_ml)