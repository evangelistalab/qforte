cmake_minimum_required(VERSION 3.4)
set (CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
#set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(qforte)

# Set source directory
set(SOURCE_DIR "src/qforte")

# Tell CMake that headers are also in SOURCE_DIR
include_directories(${SOURCE_DIR})
#set(SOURCES "${SOURCE_DIR}/math.cc")

# Generate Python module
add_subdirectory(lib/pybind11)
add_subdirectory(lib/fmt)
pybind11_add_module(qforte ${SOURCES} "${SOURCE_DIR}/bindings.cc"
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/quantum_basis.cc"
    "${SOURCE_DIR}/quantum_circuit.cc"
    "${SOURCE_DIR}/quantum_computer.cc"
    "${SOURCE_DIR}/quantum_gate.cc"
    "${SOURCE_DIR}/quantum_operator.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/sq_op_pool.cc")

target_link_libraries(qforte PRIVATE fmt-header-only)

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc")

# Generate a test executable
include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)
add_executable("${PROJECT_NAME}_test" ${TESTS})

#include_directories(${SOURCE_DIR})
#include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)

add_executable("${PROJECT_NAME}_benchmarks"
    benchmarks/benchmarks.cc
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/quantum_basis.cc"
    "${SOURCE_DIR}/quantum_circuit.cc"
    "${SOURCE_DIR}/quantum_computer.cc"
    "${SOURCE_DIR}/quantum_gate.cc"
    "${SOURCE_DIR}/quantum_operator.cc")

target_link_libraries(qforte_benchmarks PRIVATE fmt-header-only)
