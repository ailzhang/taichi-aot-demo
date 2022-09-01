#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <glad/gl.h>

#include "mpm88.hpp"
#include <signal.h>
#include <inttypes.h>

namespace demo {

namespace {
void check_opengl_error(const std::string &msg) {
  auto err = glGetError();
  if (err != GL_NO_ERROR) {
    std::cout << msg << ":" << (int)err << std::endl;
  }
}
}

constexpr int kNrParticles = 8192 * 2;
constexpr int kNGrid = 128;
constexpr size_t N_ITER = 50;

class MPM88DemoImpl {
 public:
  MPM88DemoImpl(const std::string &aot_path, TiArch arch) {
    runtime_ = ti::Runtime(arch);

    module_ = runtime_.load_aot_module(aot_path.c_str());

    // Prepare Ndarray for model
    const std::vector<uint32_t> shape_1d = {kNrParticles};
    const std::vector<uint32_t> shape_2d = {kNGrid, kNGrid};
    const std::vector<uint32_t> vec2_shape = {2};
    const std::vector<uint32_t> vec3_shape = {3};
    const std::vector<uint32_t> mat2_shape = {2, 2};

    x_ = runtime_.allocate_ndarray<float>(shape_1d, vec2_shape);
    v_ = runtime_.allocate_ndarray<float>(shape_1d, vec2_shape);
    pos_ = runtime_.allocate_ndarray<float>(shape_1d, vec3_shape);
    C_ = runtime_.allocate_ndarray<float>(shape_1d, mat2_shape);
    J_ = runtime_.allocate_ndarray<float>(shape_1d, {});
    grid_v_ = runtime_.allocate_ndarray<float>(shape_2d, vec2_shape);
    grid_m_ = runtime_.allocate_ndarray<float>(shape_2d, {});

    k_init_particles_ = module_.get_kernel("init_particles");
    k_substep_g2p_ = module_.get_kernel("substep_g2p");
    k_substep_reset_grid_ = module_.get_kernel("substep_reset_grid");
    k_substep_p2g_ = module_.get_kernel("substep_p2g");
    k_substep_update_grid_v_ = module_.get_kernel("substep_update_grid_v");

    k_init_particles_[0] = x_;
    k_init_particles_[1] = v_;
    k_init_particles_[2] = J_;

    k_substep_reset_grid_[0] = grid_v_;
    k_substep_reset_grid_[1] = grid_m_;

    k_substep_p2g_[0] = x_;
    k_substep_p2g_[1] = v_;
    k_substep_p2g_[2] = C_;
    k_substep_p2g_[3] = J_;
    k_substep_p2g_[4] = grid_v_;
    k_substep_p2g_[5] = grid_m_;

    k_substep_update_grid_v_[0] = grid_v_;
    k_substep_update_grid_v_[1] = grid_m_;

    k_substep_g2p_[0] = x_;
    k_substep_g2p_[1] = v_;
    k_substep_g2p_[2] = C_;
    k_substep_g2p_[3] = J_;
    k_substep_g2p_[4] = grid_v_;
    k_substep_g2p_[5] = pos_;

    k_init_particles_.launch();
    runtime_.wait();
  }

  void Step() {
    for (size_t i = 0; i < N_ITER; i++) {
      k_substep_reset_grid_.launch();
      k_substep_p2g_.launch();
      k_substep_update_grid_v_.launch();
      k_substep_g2p_.launch();
    }
    runtime_.wait();
    /* For debugging
    void* data = pos_.map();
    std::cout << ((float *)data)[0] << std::endl;
    pos_.unmap();
    */
  }

  TiOpenglMemoryInteropInfo x_interop() {
    TiOpenglMemoryInteropInfo info;
    ti_export_opengl_memory(runtime_, x_.memory(), &info);
    return info;
  }
 private:
  ti::Runtime runtime_;
  ti::AotModule module_;

  ti::NdArray<float> x_;
  ti::NdArray<float> v_;
  ti::NdArray<float> J_;
  ti::NdArray<float> C_;
  ti::NdArray<float> grid_v_;
  ti::NdArray<float> grid_m_;
  ti::NdArray<float> pos_;

  ti::Kernel k_init_particles_;
  ti::Kernel k_substep_reset_grid_;
  ti::Kernel k_substep_p2g_;
  ti::Kernel k_substep_update_grid_v_;
  ti::Kernel k_substep_g2p_;
};


MPM88Demo::MPM88Demo(const std::string& aot_path) {
  if (!glfwInit()) {
    return;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
  if (!window) {
    glfwTerminate();
    return;
  }

  glfwMakeContextCurrent(window);

  if (!gladLoadGL(glfwGetProcAddress)) {
    std::cout << "Failed to initialze OpenGL context" << std::endl;
    return;
  }

  int major, minor, rev;
  glfwGetVersion(&major, &minor, &rev);
  std::cout << "GLFW version: " << major << "." << minor << "." << rev << std::endl;
  std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

  // Create Taichi Device for computation
  impl_ = std::make_unique<MPM88DemoImpl>(aot_path, TiArch::TI_ARCH_OPENGL);

  //std::cout << buffer_id << std::endl;
  TiOpenglMemoryInteropInfo interop_info = impl_->x_interop();
  render_ = std::make_unique<Renderer>(interop_info.buffer, interop_info.size);
}

void MPM88Demo::Step() {
  while (!glfwWindowShouldClose(window)) {
    impl_->Step();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.0, 0.0, 0.0, 0.0);

    // Render particles
    render_->Render();

    glfwSwapBuffers(window);
    glfwPollEvents();
  }
}

MPM88Demo::~MPM88Demo() {
  impl_.reset();
}

} // namespace demo

int main(int argc, char *argv[]) {
  assert(argc == 2);
  std::string aot_path = argv[1];

  auto mpm88_demo = std::make_unique<demo::MPM88Demo>(aot_path);
  mpm88_demo->Step();

  return 0;
}
