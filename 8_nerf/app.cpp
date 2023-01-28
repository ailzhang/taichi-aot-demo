#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>
#include "assets/tutorial/hash_embedding.hpp"
#include "assets/tutorial/density_bitfield.hpp"
#include "assets/tutorial/pose.hpp"
#include "assets/tutorial/rgb_weights.hpp"
#include "assets/tutorial/sigma_weights.hpp"
#include "assets/tutorial/directions.hpp"
#include <vulkan/vulkan.h>
#include <taichi/cpp/taichi.hpp>

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
  ti::Kernel k_gen_noise_buffer_;
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
  /*
  ti::Kernel k_init_hash_embedding_;
  ti::Kernel k_init_sigma_weights_;
  ti::Kernel k_init_rgb_weights_;
  ti::Kernel k_init_density_bitfield_;
  ti::Kernel k_init_pose_;
  ti::Kernel k_init_directions_;
  */

  // ndarrays from numpy
  ti::NdArray<float> pose_;
  ti::NdArray<float> hash_embedding_;
  ti::NdArray<float> rgb_weights_;
  ti::NdArray<float> sigma_weights_;
  ti::NdArray<unsigned int> density_bitfield_;
  ti::NdArray<float> directions_;
  ti::NdArray<int> counter_;
  ti::NdArray<float> hits_t_;
  ti::NdArray<int> alive_indices_;

  // constants
  unsigned int layer1_base = 32 * 64;
  unsigned int layer2_base = layer1_base + 64 * 64;
  unsigned int grid_size = 128;
  float scale = 0.5;
  unsigned int cascades = 1;
  unsigned int N_rays = 800 * 800;

  App8_nerf() {
    runtime_ = ti::Runtime(TI_ARCH_VULKAN);
    module_ = runtime_.load_aot_module("8_nerf/assets/tutorial");
    check_taichi_error("load_aot_module failed");
    k_reset_ = module_.get_kernel("reset");
    //k_gen_noise_buffer_ = module_.get_kernel("gen_noise_buffer");
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
    //k_init_hash_embedding_ = module_.get_kernel("init_hash_embedding");
    //k_init_sigma_weights_ = module_.get_kernel("init_sigma_weights");
    //k_init_rgb_weights_ = module_.get_kernel("init_rgb_weights");
    //k_init_density_bitfield_ = module_.get_kernel("init_density_bitfield");
    //k_init_pose_ = module_.get_kernel("init_pose");
    //k_init_directions_ = module_.get_kernel("init_directions");
    check_taichi_error("get_kernel failed");
    hash_embedding_ = runtime_.allocate_ndarray<float>({}, {11445040}, /*host_accessible=*/true);
    hash_embedding_.write(hash_embedding);
    //k_init_hash_embedding_.push_arg(hash_embedding_);
    //k_init_hash_embedding_.launch();

    sigma_weights_ = runtime_.allocate_ndarray<float>({}, {layer1_base + 64 * 16}, /*host_accessible=*/true);
    sigma_weights_.write(sigma_weights);
    //k_init_sigma_weights_.push_arg(sigma_weights_);
    //k_init_sigma_weights_.launch();

    rgb_weights_ = runtime_.allocate_ndarray<float>({}, {layer2_base + 64 * 8}, /*host_accessible=*/true);
    rgb_weights_.write(rgb_weights);
    //k_init_rgb_weights_.push_arg(rgb_weights_);
    //k_init_rgb_weights_.launch();

    density_bitfield_ = runtime_.allocate_ndarray<unsigned int>({}, {cascades * grid_size * grid_size * grid_size / 32}, /*host_accessible=*/true);
    density_bitfield_.write(density_bitfield);
    //k_init_density_bitfield_.push_arg(density_bitfield_);
    //k_init_density_bitfield_.launch();

    pose_ = runtime_.allocate_ndarray<float>({3, 4}, {}, /*host_accessible=*/true);
    pose_.write(pose);
    //k_init_pose_.push_arg(pose_);
    //k_init_pose_.launch();
    // Note: For debugging
    //std::vector<unsigned int> tmp(density_bitfield.size());
    //density_bitfield_.read(tmp);
    //std::cout << "density bitfield: " << std::endl;
    //for (auto i: tmp) {
    //  std::cout << i << std::endl;
    //}

    directions_ = runtime_.allocate_ndarray<float>({1, 3}, {N_rays}, /*host_accessible=*/true);
    directions_.write(directions);
    //k_init_directions_.push_arg(directions_);
    //k_init_directions_.launch();

    counter_ = runtime_.allocate_ndarray<int>({}, {1}, /*host_accessible=*/true);

    hits_t_ = runtime_.allocate_ndarray<float>({2}, {N_rays}, /*host_accessible=*/true);
    k_fill_ndarray_.push_arg(hits_t_);
    k_fill_ndarray_.push_arg(float(-1.0));
    k_fill_ndarray_.launch();

    alive_indices_ = runtime_.allocate_ndarray<int>({}, {2 * N_rays}, /*host_accessible=*/true);

    k_init_current_index_.launch();

    runtime_.wait();
    check_taichi_error("initialization failed");
    std::cout << "Initialized!" << std::endl;
  }

  void run() {
    std::cout << "Running a frame" << std::endl;
    int samples = 0;
    int max_samples = 100;
    float T_threshold = 1e-2;
    k_reset_.push_arg(counter_);
    k_reset_.push_arg(alive_indices_);
    k_reset_.launch();

    /* Not necessary
    k_gen_noise_buffer_.launch();
    */
    k_ray_intersect_.push_arg(pose_);
    k_ray_intersect_.push_arg(directions_);
    k_ray_intersect_.push_arg(hits_t_);
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

    /*
    runtime_.wait();
    std::vector<int> tmp(2 * N_rays);
    hits_t_.read(tmp);
    std::cout << "hits_t: " << std::endl;
    for (int i = 0; i < tmp.size(); i+=2) {
      std::cout << i << ": " << tmp[i] << " " << tmp[i+1] << std::endl;;
    }
    */

      k_raymarching_test_kernel_.push_arg(counter_);
      k_raymarching_test_kernel_.push_arg(density_bitfield_);
      k_raymarching_test_kernel_.push_arg(hits_t_);
      k_raymarching_test_kernel_.push_arg(alive_indices_);
      k_raymarching_test_kernel_.push_arg(N_samples);
      k_raymarching_test_kernel_.launch();
      k_raymarching_test_kernel_.clear_args();

      k_rearange_index_.push_arg(launch_model_total);
      k_rearange_index_.launch();
      k_rearange_index_.clear_args();

      k_hash_encode_.push_arg(hash_embedding_);
      k_hash_encode_.launch();
      k_hash_encode_.clear_args();

      k_sigma_layer_.push_arg(sigma_weights_);
      k_sigma_layer_.launch();
      k_sigma_layer_.clear_args();

      k_rgb_layer_.push_arg(rgb_weights_);
      k_rgb_layer_.launch();
      k_rgb_layer_.clear_args();

      k_composite_test_.push_arg(counter_);
      k_composite_test_.push_arg(alive_indices_);
      k_composite_test_.push_arg(N_samples);
      k_composite_test_.push_arg(T_threshold);
      k_composite_test_.launch();
      k_composite_test_.clear_args();

      k_re_order_.push_arg(counter_);
      k_re_order_.push_arg(alive_indices_);
      k_re_order_.push_arg(N_alive);
      k_re_order_.launch();
      k_re_order_.clear_args();
      runtime_.wait();

      // TODO: Save NGP_rgb as image.
    }
  }
};

int main(int argc, const char** argv) {
  App8_nerf app;
  app.run();
  return 0;
}
