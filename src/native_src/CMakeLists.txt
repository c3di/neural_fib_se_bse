cmake_minimum_required(VERSION 3.18)

# project(neural-fib-se-bse-project CXX CUDA)
project(neural-fib-se-bse-project LANGUAGES CXX CUDA VERSION 1.1.0 )
set(PY_VERSION_SUFFIX "")
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

# Find the Python interpreter and the development files
if (CMAKE_CROSSCOMPILING AND NOT (APPLE AND "$ENV{CIBUILDWHEEL}" STREQUAL "1"))
    find_package(Python3 REQUIRED COMPONENTS Development.Module)
else()
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# Find the pybind11 package
include(cmake/QueryPythonForPybind11.cmake)
find_pybind11_python_first()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(GetSourceFiles)

set(FILE_TYPES *.c *.cc *.cu *.cuh *.cpp *.h *.hpp *.tcc)
find_directories(DIRS neural_fib_se_bse "${FILE_TYPES}")
generate_source_tree(LIBRARY_SOURCES "${DIRS}" "${FILE_TYPES}")

message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

set(COMPUTE_CAPABILITY "60" CACHE STRING "The CUDA compute capability that the GPU kernels should be compiled for.")

# Compile the example Python module
pybind11_add_module(_preprocess_module MODULE ${LIBRARY_SOURCES})
set_target_properties(_preprocess_module PROPERTIES
    DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}${PYTHON_MODULE_DEBUG_POSTFIX}")
target_compile_definitions(_preprocess_module PRIVATE
    MODULE_NAME=$<TARGET_FILE_BASE_NAME:_preprocess_module>
    VERSION_INFO="${PY_FULL_VERSION}"
)
target_compile_definitions(_preprocess_module PUBLIC -D_PREPROCESS_MODULE_EXPORTS)

set(PREPROCESS_MODULE_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/neural_fib_se_bse/include")
include_directories(${PREPROCESS_MODULE_INCLUDE_DIR})

target_link_libraries(_preprocess_module PUBLIC cudart)

# Hide all symbols by default (including external libraries on Linux)
set_target_properties(_preprocess_module PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN true)
if (CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_preprocess_module PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Install the Python module
install(TARGETS _preprocess_module
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})