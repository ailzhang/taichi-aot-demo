#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "assets/compiled/hash_embedding.hpp"
#include "assets/compiled/density_bitfield.hpp"
#include "assets/compiled/pose.hpp"
#include "assets/compiled/rgb_weights.hpp"
#include "assets/compiled/sigma_weights.hpp"
#include "assets/compiled/directions.hpp"
#include "assets/compiled/offsets.hpp"
#include "assets/compiled/hash_map_sizes.hpp"
#include "assets/compiled/hash_map_indicator.hpp"
#include <vulkan/vulkan.h>
#include <taichi/cpp/taichi.hpp>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_writer.h"

namespace {
void check_taichi_error(const std::string& msg) {
  TiError error = ti_get_last_error(0, nullptr);
  if (error < TI_ERROR_SUCCESS) {
    throw std::runtime_error(msg);
  }
}
}

struct App8_nerf {
  ti::Runtime runtime_;
  ti::AotModule module_;
  ti::Kernel k_reset_;
  ti::Kernel k_ray_intersect_;
  ti::Kernel k_raymarching_test_kernel_;
  ti::Kernel k_rearange_index_;
  ti::Kernel k_hash_encode_;
  ti::Kernel k_sigma_layer_;
  ti::Kernel k_rgb_layer_;
  ti::Kernel k_composite_test_;
  ti::Kernel k_re_order_;
  ti::Kernel k_fill_ndarray_;
  ti::Kernel k_init_current_index_;

  // ndarrays
  ti::NdArray<float> pose_;
  ti::NdArray<float> hash_embedding_;
  ti::NdArray<float> rgb_weights_;
  ti::NdArray<float> sigma_weights_;
  ti::NdArray<unsigned int> density_bitfield_;
  ti::NdArray<float> directions_;
  ti::NdArray<int> counter_;
  ti::NdArray<float> hits_t_;
  ti::NdArray<int> alive_indices_;
  ti::NdArray<float> opacity_;
  ti::NdArray<float> rays_o_;
  ti::NdArray<float> rays_d_;
  ti::NdArray<float> rgb_;
  ti::NdArray<int> offsets_;
  ti::NdArray<unsigned int> hash_map_sizes_;
  ti::NdArray<int> hash_map_indicator_;
  ti::NdArray<int> current_index_;
  ti::NdArray<int> model_launch_;
  ti::NdArray<int> pad_block_network_;
  ti::NdArray<float> xyzs_;
  ti::NdArray<float> dirs_;
  ti::NdArray<float> deltas_;
  ti::NdArray<float> ts_;
  ti::NdArray<float> xyzs_embedding_;
  ti::NdArray<float> final_embedding_;
  ti::NdArray<float> out_3_;
  ti::NdArray<float> out_1_;
  ti::NdArray<int> temp_hit_;
  ti::NdArray<float> occ_;
  ti::NdArray<int> run_model_ind_;
  ti::NdArray<int> N_eff_samples_;

  // constants
  unsigned int layer1_base = 32 * 64;
  unsigned int layer2_base = layer1_base + 64 * 64;
  unsigned int grid_size = 128;
  float scale = 0.5;
  unsigned int cascades = 1;
  unsigned int W = 800;
  unsigned int H = 800;
  unsigned int N_rays = W * H;
  unsigned int N_level = 16;
  unsigned int max_sample_per_ray = 1;
  unsigned int N_max_samples = N_rays * max_sample_per_ray;

  App8_nerf() {
    runtime_ = ti::Runtime(TI_ARCH_VULKAN);
    module_ = runtime_.load_aot_module("8_nerf/assets/compiled");
    check_taichi_error("load_aot_module failed");
    k_reset_ = module_.get_kernel("reset");
    k_ray_intersect_ = module_.get_kernel("ray_intersect");
    k_raymarching_test_kernel_ = module_.get_kernel("raymarching_test_kernel");
    k_rearange_index_ = module_.get_kernel("rearange_index");
    k_hash_encode_ = module_.get_kernel("hash_encode");
    k_sigma_layer_ = module_.get_kernel("sigma_layer");
    k_rgb_layer_ = module_.get_kernel("rgb_layer");
    k_composite_test_ = module_.get_kernel("composite_test");
    k_re_order_ = module_.get_kernel("re_order");
    k_fill_ndarray_ = module_.get_kernel("fill_ndarray");
    k_init_current_index_ = module_.get_kernel("init_current_index");
    check_taichi_error("get_kernel failed");

    hash_embedding_ = runtime_.allocate_ndarray<float>({11445040}, {}, /*host_accessible=*/true);
    hash_embedding_.write(hash_embedding);

    sigma_weights_ = runtime_.allocate_ndarray<float>({layer1_base + 64 * 16}, {}, /*host_accessible=*/true);
    sigma_weights_.write(sigma_weights);

    rgb_weights_ = runtime_.allocate_ndarray<float>({layer2_base + 64 * 8}, {}, /*host_accessible=*/true);
    rgb_weights_.write(rgb_weights);

    density_bitfield_ = runtime_.allocate_ndarray<unsigned int>({cascades * grid_size * grid_size * grid_size / 32}, {}, /*host_accessible=*/true);
    density_bitfield_.write(density_bitfield);

    pose_ = runtime_.allocate_ndarray<float>({}, {3, 4}, /*host_accessible=*/true);
    pose_.write(pose);

    occ_ = runtime_.allocate_ndarray<float>({N_rays}, {}, /*host_accessible=*/true);

    run_model_ind_ = runtime_.allocate_ndarray<int>({N_max_samples}, {}, /*host_accessible=*/true);
    N_eff_samples_ = runtime_.allocate_ndarray<int>({N_rays}, {}, /*host_accessible=*/true);

    directions_ = runtime_.allocate_ndarray<float>({N_rays}, {1, 3}, /*host_accessible=*/true);
    directions_.write(directions);

    counter_ = runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/true);

    hits_t_ = runtime_.allocate_ndarray<float>({N_rays}, {2}, /*host_accessible=*/true);
    k_fill_ndarray_.push_arg(hits_t_);
    k_fill_ndarray_.push_arg(float(-1.0));
    k_fill_ndarray_.launch();

    alive_indices_ = runtime_.allocate_ndarray<int>({2 * N_rays}, {}, /*host_accessible=*/true);

    current_index_ = runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/true);
    k_init_current_index_[0] = current_index_;
    k_init_current_index_.launch();

    model_launch_ = runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/true);
    pad_block_network_ = runtime_.allocate_ndarray<int>({1}, {}, /*host_accessible=*/true);

    opacity_ = runtime_.allocate_ndarray<float>({N_rays}, {}, /*host_accessible=*/true);
    rays_o_ = runtime_.allocate_ndarray<float>({N_rays}, {3}, /*host_accessible=*/true);
    rays_d_ = runtime_.allocate_ndarray<float>({N_rays}, {3}, /*host_accessible=*/true);
    rgb_ = runtime_.allocate_ndarray<float>({N_rays}, {3}, /*host_accessible=*/true);

    xyzs_ = runtime_.allocate_ndarray<float>({N_max_samples}, {3}, /*host_accessible=*/true);
    dirs_ = runtime_.allocate_ndarray<float>({N_max_samples}, {3}, /*host_accessible=*/true);
    deltas_= runtime_.allocate_ndarray<float>({N_max_samples}, {}, /*host_accessible=*/true);
    ts_ = runtime_.allocate_ndarray<float>({N_max_samples}, {}, /*host_accessible=*/true);

    xyzs_embedding_ = runtime_.allocate_ndarray<float>({N_max_samples, 32}, {}, /*host_accessible=*/true);
    final_embedding_ = runtime_.allocate_ndarray<float>({N_max_samples, 16}, {}, /*host_accessible=*/true);
    out_3_ = runtime_.allocate_ndarray<float>({N_max_samples, 3}, {}, /*host_accessible=*/true);
    out_1_ = runtime_.allocate_ndarray<float>({N_max_samples}, {}, /*host_accessible=*/true);
    temp_hit_ = runtime_.allocate_ndarray<int>({N_max_samples}, {}, /*host_accessible=*/true);

    // Hash table init
    offsets_ = runtime_.allocate_ndarray<int>({N_level}, {}, /*host_accessible=*/true);
    offsets_.write(offsets);

    hash_map_sizes_ = runtime_.allocate_ndarray<unsigned int>({N_level}, {}, /*host_accessible=*/true);
    hash_map_sizes_.write(hash_map_sizes);

    hash_map_indicator_ = runtime_.allocate_ndarray<int>({N_level}, {}, /*host_accessible=*/true);
    hash_map_indicator_.write(hash_map_indicator);

    runtime_.wait();
    check_taichi_error("initialization failed");
    std::cout << "Initialized!" << std::endl;
  }

  void run() {
    std::cout << "Running a frame" << std::endl;
    int samples = 0;
    int max_samples = 100;
    float T_threshold = 1e-4;
    k_reset_[0] = counter_;
    k_reset_[1] = alive_indices_;
    k_reset_[2] = opacity_;
    k_reset_.launch();

    k_ray_intersect_[0] = pose_;
    k_ray_intersect_[1] = directions_;
    k_ray_intersect_[2] = hits_t_;
    k_ray_intersect_[3] = rays_o_;
    k_ray_intersect_[4] = rays_d_;
    k_ray_intersect_.launch();
    runtime_.wait();

    while (samples < max_samples) {
      std::vector<int> counter{1};
      counter_.read(counter);
      int N_alive = counter[0];
      if (N_alive == 0) {
          break;
      }

      int N_samples = std::max(std::min(int(N_rays / N_alive), 64), 1);
      std::cout << "samples: " << samples << " N_alive: " << N_alive << " N_samples: " << N_samples << std::endl;
      samples += N_samples;
      int launch_model_total = N_alive * N_samples;

      k_raymarching_test_kernel_[0] = counter_;
      k_raymarching_test_kernel_[1] = density_bitfield_;
      k_raymarching_test_kernel_[2] = hits_t_;
      k_raymarching_test_kernel_[3] = alive_indices_;
      k_raymarching_test_kernel_[4] = rays_o_;
      k_raymarching_test_kernel_[5] = rays_d_;
      k_raymarching_test_kernel_[6] = current_index_;
      k_raymarching_test_kernel_[7] = xyzs_;
      k_raymarching_test_kernel_[8] = dirs_;
      k_raymarching_test_kernel_[9] = deltas_;
      k_raymarching_test_kernel_[10] = ts_;
      k_raymarching_test_kernel_[11] = run_model_ind_;
      k_raymarching_test_kernel_[12] = N_eff_samples_;
      k_raymarching_test_kernel_[13] = N_samples;
      k_raymarching_test_kernel_.launch();

      k_rearange_index_[0] = model_launch_;
      k_rearange_index_[1] = pad_block_network_;
      k_rearange_index_[2] = temp_hit_;
      k_rearange_index_[3] = run_model_ind_;
      k_rearange_index_[4] = launch_model_total;
      k_rearange_index_.launch();

      k_hash_encode_[0] = hash_embedding_;
      k_hash_encode_[1] = offsets_;
      k_hash_encode_[2] = hash_map_sizes_;
      k_hash_encode_[3] = hash_map_indicator_;
      k_hash_encode_[4] = model_launch_;
      k_hash_encode_[5] = xyzs_;
      k_hash_encode_[6] = dirs_;
      k_hash_encode_[7] = deltas_;
      k_hash_encode_[8] = xyzs_embedding_;
      k_hash_encode_[9] = temp_hit_;
      k_hash_encode_.launch();

      k_sigma_layer_[0] = sigma_weights_;
      k_sigma_layer_[1] = model_launch_;
      k_sigma_layer_[2] = pad_block_network_;
      k_sigma_layer_[3] = xyzs_embedding_;
      k_sigma_layer_[4] = final_embedding_;
      k_sigma_layer_[5] = out_1_;
      k_sigma_layer_[6] = temp_hit_;
      k_sigma_layer_.launch();


      k_rgb_layer_[0] = rgb_weights_;
      k_rgb_layer_[1] = model_launch_;
      k_rgb_layer_[2] = pad_block_network_;
      k_rgb_layer_[3] = dirs_;
      k_rgb_layer_[4] = final_embedding_;
      k_rgb_layer_[5] = out_3_;
      k_rgb_layer_[6] = temp_hit_;
      k_rgb_layer_.launch();

      /*
    runtime_.wait();
    std::vector<float> tmp(N_max_samples * 16);
    final_embedding_.read(tmp);
    std::cout << "final_embedding_: " << std::endl;
    float sum = 0.;
    for (int i = 0; i < tmp.size(); i+=1) {
        sum += tmp[i];
      std::cout << i << ": " << tmp[i] << std::endl;;
    }
    std::cout << "sum  " << sum << std::endl;
    */


      k_composite_test_[0] = counter_;
      k_composite_test_[1] = alive_indices_;
      k_composite_test_[2] = rgb_;
      k_composite_test_[3] = opacity_;
      k_composite_test_[4] = current_index_;
      k_composite_test_[5] = deltas_;
      k_composite_test_[6] = ts_;
      k_composite_test_[7] = out_3_;
      k_composite_test_[8] = out_1_;
      k_composite_test_[9] = N_eff_samples_;
      k_composite_test_[10] = N_samples;
      k_composite_test_[11] = T_threshold;
      k_composite_test_.launch();

      k_re_order_[0] = counter_;
      k_re_order_[1] = alive_indices_;
      k_re_order_[2] = current_index_;
      k_re_order_[3] = N_alive;
      k_re_order_.launch();
      runtime_.wait();

      check_taichi_error("render a frame failed");
    }

    std::vector<float> img(W * H * 3);
    rgb_.read(img);
    unsigned char data[W * H * 3];
    for (int i = 0; i < W * H * 3; i++) {
      data[i] = uint8_t(img[i] * 255);
    }
    runtime_.wait();
    stbi_write_png("out.png", W, H, /*components=*/3, data, W * 3);
  }
};

int main(int argc, const char** argv) {
  App8_nerf app;
  app.run();
  return 0;
}
