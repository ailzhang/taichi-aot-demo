#pragma once

#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "taichi/cpp/taichi.hpp"
#include "taichi/taichi_opengl.h"
#include "renderer.h"

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
  std::unique_ptr<Renderer> render_{nullptr};
  GLFWwindow *window{nullptr};
};

} // namespace demo
