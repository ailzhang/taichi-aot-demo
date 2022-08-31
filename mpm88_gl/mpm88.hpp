#pragma once

// #ifndef NO_GLAD
// #include "glad/gl.h"
// #endif
// #include "GLFW/glfw3.h"

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "taichi.hpp"

class GLFWwindow;
namespace demo {
class MPM88DemoImpl;

class MPM88Demo {
public:
  MPM88Demo(const std::string& aot_path);
  ~MPM88Demo();


  void Step();

private:
  std::unique_ptr<MPM88DemoImpl> impl_{nullptr};
  GLFWwindow *window{nullptr};
};

} // namespace demo
