set(TAICHI_NERF_DEMO_NAME "E8_nerf")
message("-- Building ${TAICHI_NERF_DEMO_NAME}")

list(APPEND TAICHI_TUTORIAL_DEMO_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/app.cpp
)
if (NOT ANDROID)
  list(APPEND TAICHI_TUTORIAL_DEMO_SRC
    ${BACKWARD_ENABLE}
  )
endif()

add_executable(${TAICHI_NERF_DEMO_NAME} ${TAICHI_TUTORIAL_DEMO_SRC})
target_include_directories(${TAICHI_NERF_DEMO_NAME} PUBLIC
  ${PROJECT_SOURCE_DIR}/external/Vulkan-Headers/include
  ${TAICHI_C_API_INSTALL_DIR}/include)
target_link_libraries(${TAICHI_NERF_DEMO_NAME} ${taichi_c_api})
set_target_properties(${TAICHI_NERF_DEMO_NAME} PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tutorial
  RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/tutorial
  RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/tutorial)

if (ANDROID)
  target_link_libraries(${TAICHI_NERF_DEMO_NAME} android log)
endif()

if (ANDROID)
  target_link_libraries(${TAICHI_NERF_DEMO_NAME} android log)
endif()

if (NOT ANDROID)
  add_backward(${TAICHI_NERF_DEMO_NAME})
endif()
# FIXME: re-generate compiled artifacts once bug is fixed on master
#execute_process(COMMAND ${PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/assets/taichi_ngp.py" --arch vulkan --aot)
