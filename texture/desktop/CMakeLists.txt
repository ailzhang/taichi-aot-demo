cmake_minimum_required(VERSION 3.13)

project(texture_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(texture_example texture.cpp)

target_compile_options(texture_example PUBLIC -Wall -Wextra -DTI_WITH_VULKAN -DTI_INCLUDED -DTI_ARCH_x64 -g)

if (NOT DEFINED ENV{TAICHI_REPO_DIR})
    message(FATAL_ERROR "TAICHI_REPO_DIR not set")
endif()

set(TAICHI_REPO_DIR $ENV{TAICHI_REPO_DIR})

target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/taichi/backends/vulkan)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/Vulkan-Headers/include/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/SPIRV-Tools/include/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/volk/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/glm/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/imgui/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/glfw/include)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/imgui/backends)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/eigen/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/spdlog/include/)
target_include_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/external/VulkanMemoryAllocator/include/)
#target_include_directories(implicit_fem PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include/)

target_link_directories(texture_example PUBLIC ${TAICHI_REPO_DIR}/build)

target_link_libraries(texture_example PUBLIC taichi_export_core)

