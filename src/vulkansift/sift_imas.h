#ifndef VKSIFT_SIFT_IMAS_H
#define VKSIFT_SIFT_IMAS_H

#include "sift_memory.h"
#include "vkenv/vulkan_device.h"

// =============================================================================
// IMAS (ASIFT) tilt simulation pipeline — GPU port of fast_imas_IPOL.
//
// Reproduces examples/asift_gpu/imas_cpu.jl + libSimuTilts/digital_tilt.cpp:
//   tilted = fproj_cubic( finvspline_col( finvspline_row(
//              GaussianBlur1D_vertical(
//                AffineWarp_rotate( input, θ ),
//                σ_aa = 0.8 * sqrt(t² - 1)
//              )))
//            ), t )
//
// Five compute shader dispatches per warp:
//   1. AffineWarp        : input_image → rotated_image  (W_rot × H_rot)
//   2. GaussianBlur1D    : rotated_image → tilted_image (vertical, same dims)
//   3. FinvsplineRow     : tilted_image (in-place IIR per row)
//   4. FinvsplineCol     : tilted_image (in-place IIR per col)
//   5. FprojCubicY       : tilted_image → rotated_image (W_rot × ⌊H_rot/t⌋)
//
// Final tilted result lands in `rotated_image` (re-used as output buffer);
// `vksift_imasReadbackTilted` copies it to a host-mapped staging buffer for
// downstream use (e.g., feeding into vksift_jl_detect).
// =============================================================================

typedef struct vksift_ImasPipeline_T
{
  vkenv_Device dev;
  vksift_SiftMemory mem;

  VkSampler sampler;

  // AffineWarp.comp pipeline — same shader as detector->affinewarp_pipeline,
  // but bound to (input_image → rotated_image) instead of
  // (blurred_input_image → warped_input_image). Reusing the shader binary so
  // we don't double up the SPIR-V.
  VkDescriptorSetLayout warp_layout;
  VkDescriptorPool      warp_pool;
  VkDescriptorSet       warp_set;
  VkPipelineLayout      warp_pipeline_layout;
  VkPipeline            warp_pipeline;

  // GaussianBlur1D.comp — vertical σ_aa pre-resample blur.
  // Bound (rotated_image → tilted_image).
  VkDescriptorSetLayout blur_layout;
  VkDescriptorPool      blur_pool;
  VkDescriptorSet       blur_set;
  VkPipelineLayout      blur_pipeline_layout;
  VkPipeline            blur_pipeline;

  // FinvsplineRow.comp / FinvsplineCol.comp — cubic B-spline coefficient IIR.
  // Single image binding each, in-place on tilted_image.
  VkDescriptorSetLayout finvspline_layout;
  VkDescriptorPool      finvspline_pool;
  VkDescriptorSet       finvspline_row_set;
  VkDescriptorSet       finvspline_col_set;
  VkPipelineLayout      finvspline_pipeline_layout;
  VkPipeline            finvspline_row_pipeline;
  VkPipeline            finvspline_col_pipeline;

  // FprojCubicY.comp — final cubic resample. Bound (tilted → rotated).
  VkDescriptorSetLayout fproj_layout;
  VkDescriptorPool      fproj_pool;
  VkDescriptorSet       fproj_set;
  VkPipelineLayout      fproj_pipeline_layout;
  VkPipeline            fproj_pipeline;

  // FprojBilinearY.comp — alternate parallel 2×2 resample. Same descriptor
  // set + pipeline layout as cubic (identical bindings + push-const struct).
  // When selected, the IIR finvspline passes are skipped.
  VkPipeline            fproj_bilinear_pipeline;

  // Host-mapped staging buffer to read back the tilted (Float32) result.
  // Sized at worst-case rotated_image extent × Float32. Mapped persistent.
  VkBuffer        readback_buffer;
  VkDeviceMemory  readback_memory;
  void           *readback_ptr;
  VkDeviceSize    readback_size;

  // Dedicated command pool + buffer; submits on detector's general queue.
  VkCommandPool    cmd_pool;
  VkCommandBuffer  cmd_buf;
  VkFence          done_fence;

  bool created;
} *vksift_ImasPipeline;

// Push constants for FinvsplineRow / FinvsplineCol (matches both shaders).
typedef struct
{
  uint32_t width;
  uint32_t height;
} FinvsplinePushConsts;

// Push constants for FprojCubicY (matches FprojCubicY.comp).
typedef struct
{
  float    t_factor;
  uint32_t output_width;
  uint32_t output_height;
  uint32_t input_width;
  uint32_t input_height;
  float    bg_value;
} FprojCubicPushConsts;

// Lazy init — called on the first IMAS dispatch. Allocates all the Vulkan
// objects. Returns NULL on failure.
vksift_ImasPipeline vksift_createImasPipeline(vkenv_Device dev, vksift_SiftMemory mem, VkSampler sampler);

void vksift_destroyImasPipeline(vksift_ImasPipeline *pipeline_ptr);

// Run the 5-dispatch IMAS chain for the given (t, θ_rad) tilt.
// Reads from mem->input_image (caller's responsibility to have uploaded the
// original image). Writes the final tilted Float32 result into the readback
// buffer; caller can read out_w × out_h Float32 pixels from readback_ptr.
//
// Returns true on success. Sets *out_w, *out_h to the actual tilted dims.
bool vksift_runImasWarp(vksift_ImasPipeline pipeline,
                        uint32_t input_w, uint32_t input_h,
                        float t_factor, float theta_rad,
                        uint32_t *out_w, uint32_t *out_h);

#endif // VKSIFT_SIFT_IMAS_H
