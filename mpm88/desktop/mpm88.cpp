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

#define NR_PARTICLES 8192
#define N_GRID 128
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
    app_config.name         = "MPM88";
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
    alloc_params.size = NR_PARTICLES * 2 * sizeof(float);
    alloc_params.usage = taichi::lang::AllocUsage::Storage;

    taichi::lang::DeviceAllocation devalloc_x = device_->allocate_memory(alloc_params);
    auto x = taichi::lang::Ndarray(devalloc_x, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {2});

    // For debugging
    //float arr[NR_PARTICLES * 2];

    taichi::lang::DeviceAllocation devalloc_v = device_->allocate_memory(alloc_params);
    auto v = taichi::lang::Ndarray(devalloc_v, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {2});

    alloc_params.size = NR_PARTICLES * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_pos = device_->allocate_memory(alloc_params);
    auto pos = taichi::lang::Ndarray(devalloc_pos, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {3});

    alloc_params.size = NR_PARTICLES * sizeof(float) * 2 * 2;
    taichi::lang::DeviceAllocation devalloc_C = device_->allocate_memory(alloc_params);
    auto C = taichi::lang::Ndarray(devalloc_C, taichi::lang::PrimitiveType::f32, {NR_PARTICLES}, {2, 2});

    alloc_params.size = NR_PARTICLES * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_J = device_->allocate_memory(alloc_params);
    auto J = taichi::lang::Ndarray(devalloc_J, taichi::lang::PrimitiveType::f32, {NR_PARTICLES});

    alloc_params.size = N_GRID * N_GRID * 2 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_grid_v = device_->allocate_memory(alloc_params);
    auto grid_v = taichi::lang::Ndarray(devalloc_grid_v, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID}, {2});

    alloc_params.size = N_GRID * N_GRID * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_grid_m = device_->allocate_memory(alloc_params);
    auto grid_m = taichi::lang::Ndarray(devalloc_grid_m, taichi::lang::PrimitiveType::f32, {N_GRID, N_GRID});


    std::unordered_map<std::string, taichi::lang::aot::IValue> args;
    args.insert({"x", taichi::lang::aot::IValue::create(x)});
    args.insert({"v", taichi::lang::aot::IValue::create(v)});
    args.insert({"J", taichi::lang::aot::IValue::create(J)});

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

    args.insert({"C", taichi::lang::aot::IValue::create(C)});
    args.insert({"grid_v", taichi::lang::aot::IValue::create(grid_v)});
    args.insert({"grid_m", taichi::lang::aot::IValue::create(grid_m)});
    args.insert({"pos", taichi::lang::aot::IValue::create(pos)});

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

    device_->dealloc_memory(devalloc_x);
    device_->dealloc_memory(devalloc_v);
    device_->dealloc_memory(devalloc_J);
    device_->dealloc_memory(devalloc_C);
    device_->dealloc_memory(devalloc_grid_v);
    device_->dealloc_memory(devalloc_grid_m);
    device_->dealloc_memory(devalloc_pos);

    return 0;
}
