/*
 * Copyright (C) 2010 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// BEGIN_INCLUDE(all)
#include <android/log.h>
#include <android_native_app_glue.h>
#include <errno.h>
#include <jni.h>

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>

#define STB_IMAGE_IMPLEMENTATION
#include "external/include/stb_image.h"
#include "taichi/backends/vulkan/aot_module_loader_impl.h"
#include "taichi/backends/vulkan/vulkan_common.h"
#include "taichi/backends/vulkan/vulkan_loader.h"
#include "taichi/backends/vulkan/vulkan_program.h"
#include "taichi/runtime/runtime.h"
#include "taichi/ui/backends/vulkan/app_context.h"
#include "taichi/ui/backends/vulkan/gui.h"
#include "taichi/ui/backends/vulkan/renderer.h"

#define LOGI(...)                                                   \
    ((void)__android_log_print(ANDROID_LOG_INFO, "native-activity", \
                               __VA_ARGS__))
#define LOGW(...)                                                   \
    ((void)__android_log_print(ANDROID_LOG_WARN, "native-activity", \
                               __VA_ARGS__))
#define LOGE(...)                                                    \
    ((void)__android_log_print(ANDROID_LOG_ERROR, "native-activity", \
                               __VA_ARGS__))

#define RENDER_RES_X 720
#define RENDER_RES_Y 1608

struct ColorImage {
    float color[4];
};

struct engine {
    struct android_app *app;

    std::unique_ptr<taichi::ui::vulkan::Renderer> renderer;
    std::unique_ptr<taichi::lang::MemoryPool> memory_pool;
    std::unique_ptr<taichi::lang::vulkan::VkRuntime> vulkan_runtime;
    std::unique_ptr<taichi::lang::aot::Module> module;
    std::shared_ptr<taichi::ui::vulkan::Gui> gui;
    taichi::lang::RuntimeContext host_ctx;

    taichi::ui::SetImageInfo setImageInfo;
    taichi::ui::FieldInfo fInfo;
    taichi::ui::FieldInfo fInfo2;

    taichi::lang::aot::Kernel *initKernel{nullptr};
    taichi::lang::aot::Kernel *stepKernel{nullptr};
    taichi::lang::aot::Kernel *blurKernel{nullptr};
    taichi::lang::RuntimeContext hostCtxStep;
    taichi::lang::RuntimeContext hostCtxBlur;

    taichi::lang::DeviceAllocation dallocAngle;
    taichi::lang::DeviceAllocation dallocBlurFactor;
    taichi::lang::DeviceAllocation dallocTextureTmp;

    taichi::lang::DeviceAllocation dallocTextureBlur;
    taichi::lang::DeviceAllocation dallocTextureClear;

    taichi::lang::DeviceAllocation dallocImageBlur;
    taichi::lang::DeviceAllocation dallocImageClear;
    bool isClear{false};
    float blurFactor{0.0};

    bool init{false};
};

static jobject getGlobalContext(JNIEnv *env) {
    // Get the instance object of Activity Thread
    jclass activityThread = env->FindClass("android/app/ActivityThread");
    jmethodID currentActivityThread =
        env->GetStaticMethodID(activityThread, "currentActivityThread",
                               "()Landroid/app/ActivityThread;");
    jobject at =
        env->CallStaticObjectMethod(activityThread, currentActivityThread);
    // Get Application, which is the global Context
    jmethodID getApplication = env->GetMethodID(
        activityThread, "getApplication", "()Landroid/app/Application;");
    jobject context = env->CallObjectMethod(at, getApplication);
    return context;
}

static std::string GetExternalFilesDir(struct engine *engine) {
    std::string ret;

    engine->app->activity->vm->AttachCurrentThread(&engine->app->activity->env,
                                                   NULL);
    JNIEnv *env = engine->app->activity->env;

    // getExternalFilesDir() - java
    jclass cls_Env = env->FindClass("android/app/NativeActivity");
    jmethodID mid = env->GetMethodID(cls_Env, "getExternalFilesDir",
                                     "(Ljava/lang/String;)Ljava/io/File;");
    jobject obj_File = env->CallObjectMethod(getGlobalContext(env), mid, NULL);
    jclass cls_File = env->FindClass("java/io/File");
    jmethodID mid_getPath =
        env->GetMethodID(cls_File, "getPath", "()Ljava/lang/String;");
    jstring obj_Path = (jstring)env->CallObjectMethod(obj_File, mid_getPath);

    ret = env->GetStringUTFChars(obj_Path, NULL);

    engine->app->activity->vm->DetachCurrentThread();

    return ret;
}

static void copyAssetsToDataDir(struct engine *engine, const char *folder) {
    const char *filename;
    auto dir =
        AAssetManager_openDir(engine->app->activity->assetManager, folder);
    std::string out_dir = GetExternalFilesDir(engine) + "/" + folder;
    std::filesystem::create_directories(out_dir);

    while ((filename = AAssetDir_getNextFileName(dir))) {
        std::ofstream out_file(out_dir + filename, std::ios::binary);
        std::string in_filepath = std::string(folder) + filename;
        AAsset *asset =
            AAssetManager_open(engine->app->activity->assetManager,
                               in_filepath.c_str(), AASSET_MODE_UNKNOWN);
        auto in_buffer = AAsset_getBuffer(asset);
        auto size = AAsset_getLength(asset);
        out_file.write((const char *)in_buffer, size);
    }
}

void loadData(taichi::lang::vulkan::VkRuntime *vulkan_runtime,
              taichi::lang::DeviceAllocation &alloc, const void *data,
              size_t size) {
    char *const deviceArrPtr =
        reinterpret_cast<char *>(vulkan_runtime->get_ti_device()->map(alloc));
    std::memcpy(deviceArrPtr, data, size);
    vulkan_runtime->get_ti_device()->unmap(alloc);
}

static int engine_init_display(struct engine *engine) {
    // Copy the assets from the AssetManager to internal storage so we can use a
    // file system path inside Taichi.
    copyAssetsToDataDir(engine, "raindrops/");
    copyAssetsToDataDir(engine, "shaders/");

    // Create the configuration for the renderer
    taichi::ui::AppConfig app_config;
    app_config.name = "AOT Loader";
    app_config.vsync = true;
    app_config.show_window = false;
    app_config.ti_arch = taichi::Arch::vulkan;
    app_config.is_packed_mode = true;
    app_config.width = ANativeWindow_getWidth(engine->app->window);
    app_config.height = ANativeWindow_getHeight(engine->app->window);
    app_config.package_path = GetExternalFilesDir(engine);

    // Create the renderer
    engine->renderer = std::make_unique<taichi::ui::vulkan::Renderer>();
    engine->renderer->init(
        nullptr, (taichi::ui::TaichiWindow *)engine->app->window, app_config);

    taichi::uint64 *result_buffer{nullptr};
    engine->memory_pool = std::make_unique<taichi::lang::MemoryPool>(
        taichi::Arch::vulkan, nullptr);
    result_buffer = (taichi::uint64 *)engine->memory_pool->allocate(
        sizeof(taichi::uint64) * taichi_result_buffer_entries, 8);

    // Create the Runtime
    taichi::lang::vulkan::VkRuntime::Params params;
    params.host_result_buffer = result_buffer;
    params.device = &(engine->renderer->app_context().device());
    engine->vulkan_runtime =
        std::make_unique<taichi::lang::vulkan::VkRuntime>(std::move(params));

    // @FIXME: On some Phones (MTK GPU for example),
    // VkExternalMemoryImageCreateInfo doesn't seem to support external
    // memory... it returns VK_ERROR_INVALID_EXTERNAL_HANDLE
    params.device->set_cap(
        taichi::lang::DeviceCapability::vk_has_external_memory, false);

    // Create the GUI and initialize a default background color
    engine->gui = std::make_shared<taichi::ui::vulkan::Gui>(
        &engine->renderer->app_context(), &engine->renderer->swap_chain(),
        (taichi::ui::TaichiWindow *)engine->app->window);

    engine->renderer->set_background_color({0.6, 0.6, 0.6});

    // Load the AOT module using the previously created Runtime
    taichi::lang::vulkan::AotModuleParams aotParams{
        app_config.package_path + "/raindrops/", engine->vulkan_runtime.get()};
    engine->module =
        taichi::lang::aot::Module::load(taichi::Arch::vulkan, aotParams);
    auto rootSize = engine->module->get_root_size();
    engine->vulkan_runtime->add_root_buffer(rootSize);

    // Retrieve kernels/fields/etc from AOT module so we can initialize our
    // runtime
    engine->initKernel = engine->module->get_kernel("init");
    engine->blurKernel = engine->module->get_kernel("blur");
    engine->stepKernel = engine->module->get_kernel("step");

    int width, height, channels;
    unsigned char *img = stbi_load(
        (app_config.package_path + "/raindrops/shanghai_pudong.png").c_str(),
        &width, &height, &channels, 4);

    // Load texture data
    std::vector<ColorImage> pixelsClear;
    float *imgClear = new float[width * height * channels]();

    for (int i = 0; i < width * height * channels; i++) {
        imgClear[i] = (float)img[i] / 255.0;
    }

    delete[] img;

    for (int i = 0; i < width * height; i++) {
        pixelsClear.push_back(ColorImage{imgClear[4 * i + 0],
                                         imgClear[4 * i + 1],
                                         imgClear[4 * i + 2], 1.0});
    }

    // Prepare Ndarray for model
    taichi::lang::Device::AllocParams allocParamsImageClear,
        allocParamsImageBlur, allocParamsTexture, allocParamsAngle,
        allocParamsBlurFactor;

    allocParamsImageBlur.size =
        RENDER_RES_X * RENDER_RES_Y * sizeof(taichi::float32) * 4;
    allocParamsImageClear.size =
        RENDER_RES_X * RENDER_RES_Y * sizeof(taichi::float32) * 4;
    allocParamsTexture.size =
        RENDER_RES_X * RENDER_RES_Y * sizeof(taichi::float32) * 4;
    allocParamsAngle.size = sizeof(taichi::float32);
    allocParamsBlurFactor.size = sizeof(taichi::float32);

    engine->dallocTextureBlur =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsTexture);
    engine->dallocTextureClear =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsTexture);
    engine->dallocTextureTmp =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsTexture);
    engine->dallocAngle =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsAngle);
    engine->dallocBlurFactor =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsBlurFactor);

    engine->dallocImageBlur =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsImageBlur);
    engine->dallocImageClear =
        engine->vulkan_runtime->get_ti_device()->allocate_memory(
            allocParamsImageClear);

    loadData(engine->vulkan_runtime.get(), engine->dallocTextureClear,
             pixelsClear.data(), sizeof(ColorImage) * pixelsClear.size());

    taichi::float32 *angleBuffer = reinterpret_cast<taichi::float32 *>(
        engine->vulkan_runtime->get_ti_device()->map(engine->dallocAngle));
    angleBuffer[0] = 0;
    engine->vulkan_runtime->get_ti_device()->unmap(engine->dallocAngle);

    memset(&engine->hostCtxStep, 0, sizeof(taichi::lang::RuntimeContext));
    memset(&engine->hostCtxBlur, 0, sizeof(taichi::lang::RuntimeContext));

    engine->host_ctx.set_arg(0, &engine->dallocImageClear);
    engine->host_ctx.set_device_allocation(0, true);
    engine->host_ctx.extra_args[0][0] = RENDER_RES_X;
    engine->host_ctx.extra_args[0][1] = RENDER_RES_Y;
    engine->host_ctx.extra_args[0][2] = 4;
    engine->host_ctx.set_arg(1, &engine->dallocTextureClear);
    engine->host_ctx.set_device_allocation(1, true);
    engine->host_ctx.extra_args[1][0] = RENDER_RES_Y;
    engine->host_ctx.extra_args[1][1] = RENDER_RES_X;
    engine->host_ctx.extra_args[1][2] = 4;
    engine->host_ctx.set_arg(2, &engine->dallocBlurFactor);
    engine->host_ctx.set_device_allocation(2, true);
    engine->host_ctx.extra_args[2][0] = 1;
    engine->host_ctx.extra_args[2][1] = 1;
    engine->host_ctx.extra_args[2][2] = 1;

    engine->hostCtxStep.set_arg(0, &engine->dallocImageBlur);
    engine->hostCtxStep.set_device_allocation(0, true);
    engine->hostCtxStep.extra_args[0][0] = RENDER_RES_X;
    engine->hostCtxStep.extra_args[0][1] = RENDER_RES_Y;
    engine->hostCtxStep.extra_args[0][2] = 4;
    engine->hostCtxStep.set_arg(1, &engine->dallocTextureBlur);
    engine->hostCtxStep.set_device_allocation(1, true);
    engine->hostCtxStep.extra_args[1][0] = RENDER_RES_Y;
    engine->hostCtxStep.extra_args[1][1] = RENDER_RES_X;
    engine->hostCtxStep.extra_args[1][2] = 4;
    engine->hostCtxStep.set_arg(2, &engine->dallocAngle);
    engine->hostCtxStep.set_device_allocation(2, true);
    engine->hostCtxStep.extra_args[2][0] = 1;
    engine->hostCtxStep.extra_args[2][1] = 1;
    engine->hostCtxStep.extra_args[2][2] = 1;
    engine->hostCtxStep.set_arg(3, &engine->dallocBlurFactor);
    engine->hostCtxStep.set_device_allocation(3, true);
    engine->hostCtxStep.extra_args[3][0] = 1;
    engine->hostCtxStep.extra_args[3][1] = 1;
    engine->hostCtxStep.extra_args[3][2] = 1;

    engine->hostCtxBlur.set_arg(0, &engine->dallocTextureClear);
    engine->hostCtxBlur.set_device_allocation(0, true);
    engine->hostCtxBlur.extra_args[0][0] = RENDER_RES_Y;
    engine->hostCtxBlur.extra_args[0][1] = RENDER_RES_X;
    engine->hostCtxBlur.extra_args[0][2] = 4;

    engine->hostCtxBlur.set_arg(1, &engine->dallocTextureTmp);
    engine->hostCtxBlur.set_device_allocation(1, true);
    engine->hostCtxBlur.extra_args[1][0] = RENDER_RES_Y;
    engine->hostCtxBlur.extra_args[1][1] = RENDER_RES_X;
    engine->hostCtxBlur.extra_args[1][2] = 4;

    engine->hostCtxBlur.set_arg(2, &engine->dallocTextureBlur);
    engine->hostCtxBlur.set_device_allocation(2, true);
    engine->hostCtxBlur.extra_args[2][0] = RENDER_RES_Y;
    engine->hostCtxBlur.extra_args[2][1] = RENDER_RES_X;
    engine->hostCtxBlur.extra_args[2][2] = 4;
    engine->hostCtxBlur.set_arg(3, &engine->dallocBlurFactor);
    engine->hostCtxBlur.set_device_allocation(3, true);
    engine->hostCtxBlur.extra_args[3][0] = 1;
    engine->hostCtxBlur.extra_args[3][1] = 1;
    engine->hostCtxBlur.extra_args[3][2] = 1;

    // Describe information to render the image with Vulkan
    engine->fInfo.field_type = taichi::ui::FieldType::Scalar;
    engine->fInfo.matrix_rows = 1;
    engine->fInfo.matrix_cols = 1;
    engine->fInfo.shape = {
        RENDER_RES_X, RENDER_RES_Y};  // Dimensions from taichi python kernels
    engine->fInfo.dtype = taichi::lang::PrimitiveType::f32;
    engine->fInfo.dev_alloc = engine->dallocImageBlur;

    engine->fInfo2.field_type = taichi::ui::FieldType::Scalar;
    engine->fInfo2.matrix_rows = 1;
    engine->fInfo2.matrix_cols = 1;
    engine->fInfo2.shape = {
        RENDER_RES_X, RENDER_RES_Y};  // Dimensions from taichi python kernels
    engine->fInfo2.dtype = taichi::lang::PrimitiveType::f32;
    engine->fInfo2.dev_alloc = engine->dallocImageClear;

    engine->initKernel->launch(&engine->host_ctx);
    engine->blurKernel->launch(&engine->hostCtxBlur);
    engine->vulkan_runtime->synchronize();

    engine->setImageInfo.img = engine->fInfo;

    engine->init = true;

    return 0;
}

/**
 * Just the current frame in the display.
 */
static void engine_draw_frame(struct engine *engine) {
    if (!engine->init) {
        // No display.
        return;
    }

    // Calculate Frame Time / FPS
    static auto t_previous_frame = std::chrono::high_resolution_clock::now();
    auto t_now_frame             = std::chrono::high_resolution_clock::now();
    double elapsed_time_ms       = std::chrono::duration<double, std::milli>(t_now_frame - t_previous_frame).count();
    t_previous_frame             = t_now_frame;
    LOGI("PERF: elapsed_time_ms=%lfms FPS=%f", elapsed_time_ms, 1000.f / elapsed_time_ms);

    // Apply blur effect every frame
    static int count = 0;
    if (count % 200 == 0) {
      taichi::float32 *blurFactorBuffer = reinterpret_cast<taichi::float32 *>(
                      engine->vulkan_runtime->get_ti_device()->map(engine->dallocBlurFactor));
      blurFactorBuffer[0] = 0.1;
      engine->vulkan_runtime->get_ti_device()->unmap(engine->dallocBlurFactor);
    }
    count++;
    engine->blurKernel->launch(&engine->hostCtxBlur);


    // Create Rain Drops effect
    engine->stepKernel->launch(&engine->hostCtxStep);
    engine->vulkan_runtime->synchronize();
    if (engine->isClear) {
        engine->setImageInfo.img = engine->fInfo2;
    } else {
        engine->setImageInfo.img = engine->fInfo;
    }

    engine->renderer->set_image(engine->setImageInfo);

    // Render elements
    engine->renderer->draw_frame(engine->gui.get());
    engine->renderer->swap_chain().surface().present_image();
    engine->renderer->prepare_for_next_frame();
}

static void engine_term_display(struct engine *engine) {
    // @TODO: to implement
}

static int32_t engine_handle_input(struct android_app *app,
                                   AInputEvent *event) {
    // Implement input with Taichi Kernel
    return 0;
}

static void engine_handle_cmd(struct android_app *app, int32_t cmd) {
    struct engine *engine = (struct engine *)app->userData;
    switch (cmd) {
        case APP_CMD_INIT_WINDOW:
            // The window is being shown, get it ready.
            if (engine->app->window != NULL) {
                engine_init_display(engine);
                engine_draw_frame(engine);
            }
            break;
        case APP_CMD_TERM_WINDOW:
            // The window is being hidden or closed, clean it up.
            engine_term_display(engine);
            break;
    }
}

void android_main(struct android_app *state) {
    struct engine engine;

    memset(&engine, 0, sizeof(engine));
    state->userData = &engine;
    state->onAppCmd = engine_handle_cmd;
    state->onInputEvent = engine_handle_input;
    engine.app = state;

    while (1) {
        // Read all pending events.
        int ident;
        int events;
        struct android_poll_source *source;

        // If not animating, we will block forever waiting for events.
        // If animating, we loop until all events are read, then continue
        // to draw the next frame of animation.
        while ((ident = ALooper_pollAll(0, NULL, &events, (void **)&source)) >=
               0) {
            // Process this event.
            if (source != NULL) {
                source->process(state, source);
            }

            // Check if we are exiting.
            if (state->destroyRequested != 0) {
                engine_term_display(&engine);
                return;
            }
        }

        engine_draw_frame(&engine);
    }
}
// END_INCLUDE(all)
