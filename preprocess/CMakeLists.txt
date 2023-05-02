cmake_minimum_required(VERSION 3.18)

project(extended_heightfield CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 1388")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --diag-suppress 1394")

message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")
message(STATUS "CMAKE_CUDA_COMPILER_ID = ${CMAKE_CUDA_COMPILER_ID}")
message(STATUS "CMAKE_CUDA_COMPILER_VERSION = ${CMAKE_CUDA_COMPILER_VERSION}")

set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${EXECUTABLE_OUTPUT_PATH})
set(CMAKE_USE_RELATIVE_PATHS FALSE)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(COMPUTE_CAPABILITY "60" CACHE STRING "The CUDA compute capability that the GPU kernels should be compiled for.")
set(CMAKE_CUDA_RUNTIME_LIBRARY STATIC)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(GetSourceFiles)

set(FILE_TYPES *.c *.cc *.cu *.cuh *.cpp *.h *.hpp *.tcc)
find_directories(DIRS extended_heightfield "${FILE_TYPES}")
generate_source_tree(LIBRARY_SOURCES "${DIRS}" "${FILE_TYPES}")

set(EXTENDED_HEIGHTFIELD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extended_heightfield/include")

find_package(Python 3.9 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

message(STATUS "Python_FOUND:${Python_FOUND}")
message(STATUS "Python_VERSION:${Python_VERSION}")
message(STATUS "Python_Development_FOUND:${Python_Development_FOUND}")
message(STATUS "Python_LIBRARIES:${Python_LIBRARIES}")

set(pybind11_DIR "C:/Python39/Lib/site-packages/pybind11/share/cmake/pybind11")
find_package(pybind11 REQUIRED)

include_directories(${Python_INCLUDE_DIRS})
include_directories(${Python_NumPy_INCLUDE_DIRS})
include_directories(${pybind11_INCLUDE_DIRS})
include_directories(${EXTENDED_HEIGHTFIELD_INCLUDE_DIR})
				  
add_library(extended_heightfield SHARED ${LIBRARY_SOURCES} )

set_target_properties(extended_heightfield PROPERTIES SUFFIX  ".pyd")
set_target_properties(extended_heightfield PROPERTIES RUNTIME_OUTPUT_DIRECTORY "../python_binding")

if(MSVC)
	target_link_libraries(extended_heightfield ${Python_LIBRARY_RELEASE})
else()
	target_link_libraries(extended_heightfield ${Python_LIBRARIES})
endif()

target_link_libraries(extended_heightfield
					  Python::NumPy					  
				      cudart
)

IF("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # if gcc version <9 we also need to link the stdc++fs library to support c++17's filesystem extension
    IF (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        target_link_libraries(${PROJECT_NAME} stdc++fs)
    ENDIF()
ENDIF()

# Copy necessary dlls to target directory
if(MSVC)
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_binding/Debug/")
	add_custom_command(
		TARGET extended_heightfield
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cudart64_110.dll"
		"${CMAKE_CURRENT_SOURCE_DIR}/python_binding/Debug/"
	)
	
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_binding/Release/")
	add_custom_command(
		TARGET extended_heightfield
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cudart64_110.dll"
		"${CMAKE_CURRENT_SOURCE_DIR}/python_binding/Release/"
	)
	
	file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python_binding/RelWithDebInfo/")
	add_custom_command(
		TARGET extended_heightfield
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy
		"${CMAKE_CURRENT_SOURCE_DIR}/dependencies/cudart64_110.dll"
		"${CMAKE_CURRENT_SOURCE_DIR}/python_binding/RelWithDebInfo/"
	)

endif(MSVC)

set_target_properties(extended_heightfield PROPERTIES CUDA_ARCHITECTURES ${COMPUTE_CAPABILITY})
set_target_properties(extended_heightfield PROPERTIES POSITION_INDEPENDENT_CODE ON)

INCLUDE_EXTERNAL_MSPROJECT(python_binding "../python_binding/python_binding.pyproj" PLATFORM "Any CPU")

add_dependencies(python_binding extended_heightfield)

add_executable(cli "cli/main.cpp" ${LIBRARY_SOURCES} )
add_dependencies(cli extended_heightfield)
