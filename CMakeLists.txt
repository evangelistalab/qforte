cmake_minimum_required(VERSION 3.4)

set (CMAKE_CXX_STANDARD 14)

# Code Coverage Configuration
#add_library(coverage_config INTERFACE)

option(CODE_COVERAGE "Enable coverage reporting" OFF)
if(CODE_COVERAGE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # Add required flags (GCC & LLVM/Clang)
  target_compile_options(coverage_config INTERFACE
    -O0        # no optimization
    -g         # generate debug info
    --coverage # sets all required flags
  )
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.13)
    target_link_options(coverage_config INTERFACE --coverage)
  else()
    target_link_libraries(coverage_config INTERFACE --coverage)
  endif()
endif(CODE_COVERAGE)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage") # enabling coverage

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
    "${SOURCE_DIR}/quantum_computer.cc"
    "${SOURCE_DIR}/quantum_gate.cc"
    "${SOURCE_DIR}/make_gate.cc")

target_link_libraries(qforte PRIVATE fmt-header-only)

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc")

# Generate a test executable
include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)
add_executable("${PROJECT_NAME}_test" ${TESTS})
