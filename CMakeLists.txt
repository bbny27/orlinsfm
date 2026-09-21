cmake_minimum_required(VERSION 3.20)
project(orlin_sfm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED NO_MODULE)
if(POLICY CMP0167)
  cmake_policy(SET CMP0167 OLD)  # Also support unpacked Boost header distributions.
endif()
find_package(Boost 1.75 REQUIRED)

add_library(orlin_sfm INTERFACE)
target_include_directories(orlin_sfm INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(orlin_sfm INTERFACE Eigen3::Eigen Boost::boost)

if(MSVC)
  target_compile_options(orlin_sfm INTERFACE /W4 /permissive-)
else()
  target_compile_options(orlin_sfm INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(orlin_sfm_cli src/main.cpp)
target_link_libraries(orlin_sfm_cli PRIVATE orlin_sfm)
set_target_properties(orlin_sfm_cli PROPERTIES OUTPUT_NAME orlin_sfm)

add_executable(orlin_benchmark src/benchmark.cpp)
target_link_libraries(orlin_benchmark PRIVATE orlin_sfm)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 COMPONENTS Interpreter QUIET)
  if(Python3_Interpreter_FOUND)
    add_test(NAME crosscheck COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_crosscheck.py)
    add_test(NAME crop_tool COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_crop_tool.py)
    add_test(NAME varied_datasets COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_varied_datasets.py)
  endif()
  foreach(name IN ITEMS numeric incremental_rref orlin double reduce benchmarks)
    add_executable(test_${name} tests/test_${name}.cpp)
    target_link_libraries(test_${name} PRIVATE orlin_sfm)
    add_test(NAME ${name} COMMAND test_${name})
  endforeach()
  target_compile_definitions(
    test_benchmarks PRIVATE ORLIN_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
  )
endif()
