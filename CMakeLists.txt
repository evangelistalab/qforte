cmake_minimum_required(VERSION 3.4)
set (CMAKE_CXX_STANDARD 14)
# set(CMAKE_CXX_STANDARD_REQUIRED OFF)

# optionally use OpenMP, if so use OpenMP compatible compiler
OPTION (USE_OpenMP "Use OpenMP to enamble <omp.h>" OFF)

if(USE_OpenMP)
    set(CMAKE_C_COMPILER "/usr/local/Cellar/llvm/10.0.1/bin/clang")
    set(CMAKE_CXX_COMPILER "/usr/local/Cellar/llvm/10.0.1/bin/clang++")
    set(OPENMP_LIBRARIES "/usr/local/Cellar/llvm/10.0.1/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/10.0.1/include")
    # if(CMAKE_C_COMPILER_ID MATCHES "Clang")
    #     set(OpenMP_C "${CMAKE_C_COMPILER}")
    #     set(OpenMP_C_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    #     set(OpenMP_C_LIB_NAMES "libomp" "libgomp" "libiomp5")
    #     set(OpenMP_libomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    #     set(OpenMP_libgomp_LIBRARY ${OpenMP_C_LIB_NAMES})
    #     set(OpenMP_libiomp5_LIBRARY ${OpenMP_C_LIB_NAMES})
    # endif()
    # if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    #   set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
    #   set(OpenMP_CXX_FLAGS "-fopenmp=libomp -Wno-unused-command-line-argument")
    #   set(OpenMP_CXX_LIB_NAMES "libomp" "libgomp" "libiomp5")
    #   set(OpenMP_libomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    #   set(OpenMP_libgomp_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    #   set(OpenMP_libiomp5_LIBRARY ${OpenMP_CXX_LIB_NAMES})
    # endif()
endif()

# Need to check how much of this is actually needed!

# Find BLAS library
cmake_policy(SET CMP0074 NEW)

# Bypass FindBLAS module
set(BLA_VENDOR "Generic")

# (Nick) I need the following, may be elsewhere or unneeded on other os. 
# set(CMAKE_C_COMPILER "/usr/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/bin/clang++")

# Use these if installing to anaconda
set(CMAKE_PREFIX_PATH "/home/tylerpbrady/anaconda3/envs/qf2")
set(BLAS_LIBRARIES "${CMAKE_PREFIX_PATH}/lib") 
set(CBLAS_INCLUDE_DIR "${CMAKE_PREFIX_PATH}/include") #need
set(OPENBLAS_EXE "${CMAKE_PREFIX_PATH}/lib/libopenblas.so") #need, may be libopenblas.so for linux
# find_package(BLAS REQUIRED)

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
    "${SOURCE_DIR}/qubit_basis.cc"
    "${SOURCE_DIR}/circuit.cc"
    "${SOURCE_DIR}/computer.cc"
    "${SOURCE_DIR}/fci_computer.cc"
    "${SOURCE_DIR}/fci_graph.cc"
    "${SOURCE_DIR}/tensor.cc"
    "${SOURCE_DIR}/tensor_einsum.cc"
    "${SOURCE_DIR}/tensor_operator.cc"
    "${SOURCE_DIR}/gate.cc"
    "${SOURCE_DIR}/blas_math.cc" 
    "${SOURCE_DIR}/qubit_operator.cc"
    "${SOURCE_DIR}/qubit_op_pool.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/sq_op_pool.cc"
    "${SOURCE_DIR}/sparse_tensor.cc"
    "${SOURCE_DIR}/timer.cc")

link_directories(${BLAS_LIBRARIES})
target_include_directories(qforte PRIVATE ${CBLAS_INCLUDE_DIR})
target_link_libraries(qforte PRIVATE fmt-header-only ${OPENBLAS_EXE})

# specify open MP flags
if(USE_OpenMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(qforte PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif(USE_OpenMP)

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cc"
    "${TEST_DIR}/test_math.cc")

# Generate a test executable
include_directories(${CBLAS_INCLUDE_DIR})
include_directories(lib/catch2/single_include/catch2)
include_directories(lib/fmt/include)

add_executable("${PROJECT_NAME}_test" ${TESTS})

add_executable("${PROJECT_NAME}_benchmarks"
    benchmarks/benchmarks.cc
    "${SOURCE_DIR}/helpers.cc"
    "${SOURCE_DIR}/make_gate.cc"
    "${SOURCE_DIR}/qubit_basis.cc"
    "${SOURCE_DIR}/circuit.cc"
    "${SOURCE_DIR}/computer.cc"
    "${SOURCE_DIR}/fci_computer.cc"
    "${SOURCE_DIR}/fci_graph.cc"
    "${SOURCE_DIR}/tensor.cc"
    "${SOURCE_DIR}/tensor_einsum.cc"
    "${SOURCE_DIR}/tensor_operator.cc"
    "${SOURCE_DIR}/gate.cc"
    "${SOURCE_DIR}/blas_math.cc"
    "${SOURCE_DIR}/qubit_operator.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/qubit_op_pool.cc"
    "${SOURCE_DIR}/sq_operator.cc"
    "${SOURCE_DIR}/sparse_tensor.cc"
    "${SOURCE_DIR}/timer.cc")

target_link_libraries(qforte_benchmarks PRIVATE fmt-header-only ${OPENBLAS_EXE})
