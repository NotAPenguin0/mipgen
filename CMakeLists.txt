cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(mipgen C CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Vulkan REQUIRED)

add_library(mipgen)
target_sources(mipgen PRIVATE src/mipgen.cpp)
target_include_directories(mipgen PUBLIC include)
target_include_directories(mipgen PRIVATE Vulkan::Vulkan)
target_link_libraries(mipgen PUBLIC Vulkan::Vulkan)

option(MIPGEN_NO_COMPUTE OFF)

if (NOT ${MIPGEN_NO_COMPUTE})
	target_compile_definitions(mipgen PRIVATE -DMIPGEN_COMPUTE=1)
	add_custom_command(
		OUTPUT ${CMAKE_BINARY_DIR}/mipgen.spv
		COMMAND glslc src/mipgen.comp "-o${CMAKE_BINARY_DIR}/mipgen.spv"
		DEPENDS src/mipgen.comp
		WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
	)
	add_custom_target(mipgen-shaders
		ALL DEPENDS ${CMAKE_BINARY_DIR}/mipgen.spv
		COMMENT "Mipgen - building shader"
	)
	add_dependencies(mipgen mipgen-shaders)
else() 
	target_compile_definitions(mipgen PRIVATE -DMIPGEN_COMPUTE=0)
endif()

if (CMAKE_BUILD_TYPE STREQUAL Debug OR CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo)
	target_compile_definitions(mipgen PRIVATE -DMIPGEN_DEBUG=1)
else()
	target_compile_definitions(mipgen PRIVATE -DMIPGEN_DEBUG=0)
endif()