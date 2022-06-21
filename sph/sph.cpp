#include <signal.h>
#include <iostream>
#include <chrono>

#include <taichi/runtime/program_impls/vulkan/vulkan_program.h>
#include <taichi/backends/vulkan/vulkan_common.h>
#include <taichi/backends/vulkan/vulkan_loader.h>
#include <taichi/runtime/gfx/aot_module_loader_impl.h>
#include <taichi/aot/graph_data.h>
#include <inttypes.h>

#include <taichi/gui/gui.h>
#include <taichi/ui/backends/vulkan/renderer.h>

#define NR_PARTICLES 8000
void get_data(
    taichi::lang::gfx::GfxRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  TI_ASSERT(device_arr_ptr);
  std::memcpy(data, device_arr_ptr, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}
void set_data(
    taichi::lang::gfx::GfxRuntime *vulkan_runtime,
    taichi::lang::DeviceAllocation &alloc,
    void *data,
    size_t size) {
  char *const device_arr_ptr =
      reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
  TI_ASSERT(device_arr_ptr);
  std::memcpy(device_arr_ptr, data, size);
  vulkan_runtime->get_ti_device()->unmap(alloc);
}
#include <unistd.h>
int main() {
    // Init gl window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(512, 512, "Taichi show", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Create a GGUI configuration
    taichi::ui::AppConfig app_config;
    app_config.name         = "SPH";
    app_config.width        = 512;
    app_config.height       = 512;
    app_config.vsync        = true;
    app_config.show_window  = false;
    app_config.package_path = "../"; // make it flexible later
    app_config.ti_arch      = taichi::Arch::vulkan;

    // Create GUI & renderer
    auto renderer  = std::make_unique<taichi::ui::vulkan::Renderer>();
    renderer->init(nullptr, window, app_config);

    // Initialize our Vulkan Program pipeline
    taichi::uint64 *result_buffer{nullptr};
    auto memory_pool = std::make_unique<taichi::lang::MemoryPool>(taichi::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)memory_pool->allocate(sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);
    // Create Taichi Device for computation
    taichi::lang::vulkan::VulkanDevice *device_ = &(renderer->app_context().device());
    // Create Vulkan runtime
    taichi::lang::gfx::GfxRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = device_;
    auto vulkan_runtime =
      std::make_unique<taichi::lang::gfx::GfxRuntime>(std::move(params));


    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
    taichi::lang::gfx::AotModuleParams mod_params;
    mod_params.module_path = "../shaders/";
    mod_params.runtime = vulkan_runtime.get();
    std::unique_ptr<taichi::lang::aot::Module> module = taichi::lang::aot::Module::load(taichi::Arch::vulkan, mod_params);

    auto root_size = module->get_root_size();
    printf("root buffer size=%ld\n", root_size);
    vulkan_runtime->add_root_buffer(root_size);

    auto g_init = module->get_graph("init");
    auto g_update = module->get_graph("update");


    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.host_write = false;
    alloc_params.host_read = false;
    alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;

    alloc_params.size = NR_PARTICLES * sizeof(int);
    alloc_params.host_write = alloc_params.host_read = true;
    taichi::lang::DeviceAllocation devalloc_N = device_->allocate_memory(alloc_params);
    auto N = taichi::lang::Ndarray(devalloc_N, taichi::lang::PrimitiveType::i32, {NR_PARTICLES});
    alloc_params.host_write = alloc_params.host_read = false;

    alloc_params.size = NR_PARTICLES * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_den = device_->allocate_memory(alloc_params);
    auto den = taichi::lang::Ndarray(devalloc_den, taichi::lang::PrimitiveType::f32, {NR_PARTICLES});
    taichi::lang::DeviceAllocation devalloc_pre = device_->allocate_memory(alloc_params);
    auto pre = taichi::lang::Ndarray(devalloc_pre, taichi::lang::PrimitiveType::f32, {NR_PARTICLES});

    alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_pos = device_->allocate_memory(alloc_params);
    auto pos = taichi::lang::Ndarray(devalloc_pos, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    taichi::lang::DeviceAllocation devalloc_vel = device_->allocate_memory(alloc_params);
    auto vel = taichi::lang::Ndarray(devalloc_vel, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    taichi::lang::DeviceAllocation devalloc_acc = device_->allocate_memory(alloc_params);
    auto acc = taichi::lang::Ndarray(devalloc_acc, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    alloc_params.host_write = alloc_params.host_read = true;
    taichi::lang::DeviceAllocation devalloc_boundary_box = device_->allocate_memory(alloc_params);
    auto boundary_box = taichi::lang::Ndarray(devalloc_boundary_box, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    taichi::lang::DeviceAllocation devalloc_spawn_box = device_->allocate_memory(alloc_params);
    auto spawn_box = taichi::lang::Ndarray(devalloc_spawn_box, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    taichi::lang::DeviceAllocation devalloc_gravity = device_->allocate_memory(alloc_params);
    auto gravity = taichi::lang::Ndarray(devalloc_gravity, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});
    alloc_params.host_write = alloc_params.host_read = false;


    // Initialize necessary data
    float* boundary_box_data = new float[6]{0.0, 0.0, 0.0, 1.0, 1.0, 1.0};
    float* spawn_box_data = new float[6]{0.3, 0.3, 0.3, 0.7, 0.7, 0.7};
    int* N_data = new int[3]{20, 20, 20};
    set_data(vulkan_runtime.get(), devalloc_boundary_box, (void*)boundary_box_data, 6*sizeof(float));
    set_data(vulkan_runtime.get(), devalloc_spawn_box, (void*)spawn_box_data, 6*sizeof(float));
    set_data(vulkan_runtime.get(), devalloc_N, (void*)N_data, 3*sizeof(int));
    delete[] boundary_box_data;
    delete[] spawn_box_data;
    delete[] N_data;


    std::unordered_map<std::string, taichi::lang::aot::IValue> args;
    args.insert({"pos", taichi::lang::aot::IValue::create(pos)});
    args.insert({"spawn_box", taichi::lang::aot::IValue::create(spawn_box)});
    args.insert({"N", taichi::lang::aot::IValue::create(N)});
    args.insert({"gravity", taichi::lang::aot::IValue::create(gravity)});

    g_init->run(args);
    vulkan_runtime->synchronize();

    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);


    // Describe information to render the circle with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Scalar;
    f_info.matrix_rows  = 1;
    f_info.matrix_cols  = 1;
    f_info.shape        = {NR_PARTICLES};
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = devalloc_pos;
    taichi::ui::CirclesInfo circles;
    circles.renderable_info.has_per_vertex_color = false;
    circles.renderable_info.vbo_attrs = taichi::ui::VertexAttributes::kPos;
    circles.renderable_info.vbo                  = f_info;
    circles.color                                = {0.8, 0.4, 0.1};
    circles.radius                               = 0.005f; // 0.0015f looks unclear on desktop

    renderer->set_background_color({0.6, 0.6, 0.6});

    args.insert({"pos", taichi::lang::aot::IValue::create(pos)});
    args.insert({"den", taichi::lang::aot::IValue::create(den)});
    args.insert({"pre", taichi::lang::aot::IValue::create(pre)});
    args.insert({"vel", taichi::lang::aot::IValue::create(vel)});
    args.insert({"acc", taichi::lang::aot::IValue::create(acc)});
    args.insert({"gravity", taichi::lang::aot::IValue::create(gravity)});
    args.insert({"boundary_box", taichi::lang::aot::IValue::create(boundary_box)});

    // sleep(10);
    int count = 0;
    while (!glfwWindowShouldClose(window)) {
        g_update->run(args);
        vulkan_runtime->synchronize();

        // Render elements
        renderer->circles(circles);
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    device_->dealloc_memory(devalloc_N);
    device_->dealloc_memory(devalloc_den);
    device_->dealloc_memory(devalloc_pre);
    device_->dealloc_memory(devalloc_pos);
    device_->dealloc_memory(devalloc_vel);
    device_->dealloc_memory(devalloc_acc);
    device_->dealloc_memory(devalloc_boundary_box);
    device_->dealloc_memory(devalloc_spawn_box);
    device_->dealloc_memory(devalloc_gravity);

    vulkan_runtime.reset();
    renderer->cleanup();

    return 0;
}
