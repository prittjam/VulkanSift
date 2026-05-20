#ifndef VKSIFT_SIFT_IMAS_H
#define VKSIFT_SIFT_IMAS_H

#include "sift_memory.h"
#include "sift_warp_ubo.h"
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
  // set + pipeline layout as cubic (identical bindings + UBO at set=1).
  // When selected, the IIR finvspline passes are skipped.
  VkPipeline            fproj_bilinear_pipeline;

  // Per-slot WarpParamsUBO descriptor set (bound at set=1, binding=0 in every
  // IMAS-chain shader after Phase B-2). Layout is shared across all 5
  // pipelines (same UBO struct on the GLSL side); the descriptor pool is
  // sized to allocate one set per pyramid slot, so phase C parallel waves
  // can each bind their own slot's UBO without rewriting descriptors.
  // Phase B-2 only consumes warp_ubo_desc_sets[0].
  VkDescriptorSetLayout warp_ubo_desc_set_layout;
  VkDescriptorPool      warp_ubo_desc_pool;
  VkDescriptorSet       warp_ubo_desc_sets[VKSIFT_MAX_PYRAMID_SLOTS];

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

// (Phase B-2 removed FinvsplinePushConsts / FprojCubicPushConsts /
// ImasAffineWarpPushConsts / ImasGaussBlur1DPushConsts — the IMAS-chain
// shaders now read all warp parameters from the WarpParamsUBO bound at
// set = 1, binding = 0. Build per-warp values into a WarpParamsUBO struct
// and memcpy them into mem->slots[slot].warp_params_ubo_ptr instead.)

// Lazy init — called on the first IMAS dispatch. Allocates all the Vulkan
// objects. Returns NULL on failure.
vksift_ImasPipeline vksift_createImasPipeline(vkenv_Device dev, vksift_SiftMemory mem, VkSampler sampler);

void vksift_destroyImasPipeline(vksift_ImasPipeline *pipeline_ptr);

// Refresh the IMAS pipeline's set-0 image-view bindings. The IMAS shaders'
// descriptor sets reference per-slot rotated_image/tilted_image image views
// + the single-instance cached_input_image_view; when
// vksift_prepareSiftMemoryForDetection resizes the per-slot images on a
// canvas change, the old image views are destroyed and the descriptor sets
// become stale. Callers must invoke this whenever memory_layout_updated is
// reported, BEFORE submitting any cmd buffer that uses an IMAS pipeline.
void vksift_imasRefreshDescriptorSets(vksift_ImasPipeline pipeline);

// Phase B-3 helper: compute the IMAS rotated canvas extent (W_rot × H_rot)
// for input dims (nx, ny) at rotation angle (ca = cos θ, sa = sin θ). Also
// returns the offset of the rotated-canvas origin in input-image coords
// (xmin, ymin) so callers can derive the inverse-rotation affine matrix
// matching the IMAS AffineWarp shader's convention. Exposed for use by
// vksift_dispatchFusedImasWarpForSlot (sift_detector.c), which builds the
// WarpParamsUBO + SlotDispatchBuffer in one place.
void vksift_imasComputeRotatedCanvas(uint32_t nx, uint32_t ny, float ca, float sa,
                                     int *xmin, int *xmax, int *ymin, int *ymax,
                                     uint32_t *sx, uint32_t *sy);

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
