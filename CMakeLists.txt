cmake_minimum_required(VERSION 3.28)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(srkf)

include(FetchContent)

file(GLOB_RECURSE cpp_src src/*.cpp)
add_executable(srkf ${cpp_src})

if(NOT MSVC)
    target_compile_options(srkf PRIVATE -Wall -Wextra -pedantic)
endif()
set_property(TARGET srkf PROPERTY COMPILE_WARNING_AS_ERROR ON)

target_compile_features(srkf PUBLIC cxx_std_23)
target_include_directories(srkf PRIVATE src)

FetchContent_Declare(
    Eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 5.0.1
)
FetchContent_MakeAvailable(Eigen3)
target_link_libraries(srkf PUBLIC Eigen3::Eigen)
