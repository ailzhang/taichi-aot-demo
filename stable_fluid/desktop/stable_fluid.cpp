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

#define NX 512
#define NY 512
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

float randn() {
  return static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

#include <unistd.h>
int main() {
    // Init gl window
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWwindow* window = glfwCreateWindow(NX, NY, "Taichi show", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Create a GGUI configuration
    taichi::ui::AppConfig app_config;
    app_config.name         = "MPM88";
    app_config.width        = NX;
    app_config.height       = NY;
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

    auto g1= module->get_graph("g1");
    auto g2 = module->get_graph("g2");


    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams alloc_params;
    alloc_params.host_write = false;
    alloc_params.host_read = false;
    alloc_params.usage = taichi::lang::AllocUsage::Storage;
    alloc_params.size = NX * NY * 2 * sizeof(float);

    taichi::lang::DeviceAllocation devalloc_v = device_->allocate_memory(alloc_params);
    auto v = taichi::lang::Ndarray(devalloc_v, taichi::lang::PrimitiveType::f32, {NX, NY}, {2});

    taichi::lang::DeviceAllocation devalloc_new_v = device_->allocate_memory(alloc_params);
    auto new_v = taichi::lang::Ndarray(devalloc_new_v, taichi::lang::PrimitiveType::f32, {NX, NY}, {2});

    alloc_params.size = NX * NY * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_v_div = device_->allocate_memory(alloc_params);
    auto v_div = taichi::lang::Ndarray(devalloc_v_div, taichi::lang::PrimitiveType::f32, {NX, NY});

    alloc_params.size = NX * NY * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_pressure = device_->allocate_memory(alloc_params);
    auto pressure = taichi::lang::Ndarray(devalloc_pressure, taichi::lang::PrimitiveType::f32, {NX, NY});

    taichi::lang::DeviceAllocation devalloc_new_pressure = device_->allocate_memory(alloc_params);
    auto new_pressure = taichi::lang::Ndarray(devalloc_new_pressure, taichi::lang::PrimitiveType::f32, {NX, NY});

    alloc_params.size = NX * NY * 3 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_dye = device_->allocate_memory(alloc_params);
    auto dye = taichi::lang::Ndarray(devalloc_dye, taichi::lang::PrimitiveType::f32, {NX, NY}, {3});

    taichi::lang::DeviceAllocation devalloc_new_dye = device_->allocate_memory(alloc_params);
    auto new_dye = taichi::lang::Ndarray(devalloc_new_dye, taichi::lang::PrimitiveType::f32, {NX, NY}, {3});

    alloc_params.host_write = true;
    alloc_params.size = 8 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_mouse_data = device_->allocate_memory(alloc_params);
    auto mouse_data = taichi::lang::Ndarray(devalloc_mouse_data, taichi::lang::PrimitiveType::f32, {8});

    alloc_params.host_write = false;
    alloc_params.size = NX * NY * 4 * sizeof(float);
    taichi::lang::DeviceAllocation devalloc_dye_image = device_->allocate_memory(alloc_params);
    auto dye_image = taichi::lang::Ndarray(devalloc_dye_image, taichi::lang::PrimitiveType::f32, {NX, NY}, {4});

    // For debugging
    //float arr[NR_PARTICLES * 2];
    // Create a GUI even though it's not used in our case (required to
    // render the renderer)
    auto gui = std::make_shared<taichi::ui::vulkan::Gui>(&renderer->app_context(), &renderer->swap_chain(), window);


    // Describe information to render the circle with Vulkan
    taichi::ui::FieldInfo f_info;
    f_info.valid        = true;
    f_info.field_type   = taichi::ui::FieldType::Scalar;
    f_info.matrix_rows  = 1;
    f_info.matrix_cols  = 1;
    f_info.shape        = {NX, NY};
    f_info.field_source = taichi::ui::FieldSource::TaichiVulkan;
    f_info.dtype        = taichi::lang::PrimitiveType::f32;
    f_info.snode        = nullptr;
    f_info.dev_alloc    = devalloc_dye_image;

    taichi::ui::SetImageInfo set_image_info;
    set_image_info.img = f_info;

    renderer->set_background_color({0.6, 0.6, 0.6});

    std::unordered_map<std::string, taichi::lang::aot::IValue> args;
    args.insert({"mouse_data", taichi::lang::aot::IValue::create(mouse_data)});
    args.insert({"velocities_pair_cur", taichi::lang::aot::IValue::create(v)});
    args.insert({"velocities_pair_nxt", taichi::lang::aot::IValue::create(new_v)});
    args.insert({"dyes_pair_cur", taichi::lang::aot::IValue::create(dye)});
    args.insert({"dyes_pair_nxt", taichi::lang::aot::IValue::create(new_dye)});
    args.insert({"pressures_pair_cur", taichi::lang::aot::IValue::create(pressure)});
    args.insert({"pressures_pair_nxt", taichi::lang::aot::IValue::create(new_pressure)});
    args.insert({"velocity_divs", taichi::lang::aot::IValue::create(v_div)});
    args.insert({"dye_image", taichi::lang::aot::IValue::create(dye_image)});

    bool swap = true;

    while (!glfwWindowShouldClose(window)) {
        // Generate user inputs location randomly
        // Directions and colors are hardcoded here.
        float x_pos = randn() * NX;
        float y_pos = randn() * NY;
        float direction_x = randn();
        float direction_y = randn();
        direction_x = direction_x / sqrt(direction_x * direction_x + direction_y * direction_y);
        direction_y = direction_y / sqrt(direction_x * direction_x + direction_y * direction_y);
        float r = randn();
        float g = randn();
        float b = randn();

        float pos_data[8] = {direction_x, direction_y, x_pos, y_pos, r, g, b, 0.0};
        set_data(vulkan_runtime.get(), devalloc_mouse_data, (void*)pos_data, 8*sizeof(float));

        if (swap) {
            g1->run(args);
            swap = false;
        } else {
            g2->run(args);
            swap = true;
        }

        vulkan_runtime->synchronize();

        // Render elements
        renderer->set_image(set_image_info);
        renderer->draw_frame(gui.get());
        renderer->swap_chain().surface().present_image();
        renderer->prepare_for_next_frame();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    device_->dealloc_memory(devalloc_v);
    device_->dealloc_memory(devalloc_new_v);
    device_->dealloc_memory(devalloc_v_div);
    device_->dealloc_memory(devalloc_pressure);
    device_->dealloc_memory(devalloc_new_pressure);
    device_->dealloc_memory(devalloc_dye);
    device_->dealloc_memory(devalloc_new_dye);
    device_->dealloc_memory(devalloc_mouse_data);
    device_->dealloc_memory(devalloc_dye_image);

    return 0;
}
