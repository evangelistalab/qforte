cmake_minimum_required(VERSION 2.8.12)
set (CMAKE_CXX_STANDARD 11)

project(qforte)

# Set source directory
set(SOURCE_DIR "src/qforte")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
set(SOURCES "${SOURCE_DIR}/math.cc")

# Generate Python module
add_subdirectory(lib/pybind11)
pybind11_add_module(qforte ${SOURCES} "${SOURCE_DIR}/bindings.cc")

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc")

# Generate a test executable
include_directories(lib/catch2/single_include/catch2)
add_executable("${PROJECT_NAME}_test" ${TESTS})
