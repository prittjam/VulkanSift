#ifndef VKSIFT_SIFTDETECTOR
#define VKSIFT_SIFTDETECTOR

#include "sift_memory.h"
#include "sift_warp_ubo.h"
#include "vulkansift/vulkansift_types.h"

#include "vkenv/vulkan_device.h"

#define VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE 20u

typedef struct vksift_SiftDetector_T
{
  vkenv_Device dev;      // parent device
  vksift_SiftMemory mem; // associated memory

  // Current buffer target for the SIFT detector
  // This defines where found features will be stored
  uint32_t curr_buffer_idx;

  VkQueue general_queue;
  VkQueue async_ownership_transfer_queue;

  VkCommandPool general_command_pool;
  VkCommandPool async_transfer_command_pool;

  VkCommandBuffer detection_command_buffer;
  // Variant of detection_command_buffer that skips the staging→input_image
  // upload and instead reads input_image from a device-side R32F source
  // (mem->rotated_image, the IMAS pipeline's output) via the quantize shader.
  // Used by vksift_detectFeaturesOnImas() — no host roundtrip.
  VkCommandBuffer detection_command_buffer_from_imas;
  // Phase B-3: fully-fused IMAS-chain + Quantize + SIFT detect command buffer,
  // pre-recorded once per pyramid slot. Combines the work that previously ran
  // on the IMAS pipeline's cmd buffer (run_imas) AND
  // detection_command_buffer_from_imas into a single GPU submission per warp,
  // eliminating one fence wait. IMAS-chain dispatch dimensions come from the
  // slot's indirect-dispatch buffer (mem->slots[s].dispatch_buffer); affine
  // matrix + canvas dims come from mem->slots[s].warp_params_ubo. Host updates
  // both buffers before each submission and reuses the recording.
  VkCommandBuffer fused_imas_detect_command_buffer[VKSIFT_MAX_PYRAMID_SLOTS];

  // Phase B-3: external reference to the lazy-created IMAS pipeline (owned by
  // vksift_Instance_T, set by vulkansift.c when the pipeline first comes up).
  // The fused-cmd-buffer recording needs to bind the IMAS shaders' pipelines
  // and descriptor sets; without this pointer the detector cannot record the
  // IMAS chain. NULL until first use of the fused path.
  struct vksift_ImasPipeline_T *imas_pipeline_ref;
  VkCommandBuffer acquire_buffer_ownership_command_buffer;
  VkCommandBuffer release_buffer_ownership_command_buffer;

  VkSampler image_sampler;

  // Sync objects
  VkFence end_of_detection_fence;
  VkSemaphore end_of_detection_semaphore;
  VkSemaphore buffer_ownership_released_by_transfer_semaphore;

  bool debug_marker_supported;
  PFN_vkCmdDebugMarkerBeginEXT vkCmdDebugMarkerBeginEXT;
  PFN_vkCmdDebugMarkerEndEXT vkCmdDebugMarkerEndEXT;

  // Shared WarpParamsUBO descriptor set (set = 1 in AffineWarp.comp and
  // QuantizeF32ToInput.comp). One layout, one pool, one descriptor set per
  // pyramid slot — the set's binding 0 points at mem->slots[s].warp_params_ubo.
  // Phase B-2 only consumes warp_ubo_desc_sets[0]; the rest are wired up so
  // the parallel-wave dispatcher (Phase C) can bind a different slot's UBO
  // per submitted command buffer without rewriting descriptors.
  VkDescriptorSetLayout warp_ubo_desc_set_layout;
  VkDescriptorPool warp_ubo_desc_pool;
  VkDescriptorSet warp_ubo_desc_sets[VKSIFT_MAX_PYRAMID_SLOTS];

  // Gaussian kernels
  uint32_t *gaussian_kernel_sizes;
  float *gaussian_kernels;

  // PreBlur1D set — Morel-Yu σ_aa anti-alias 1D Gaussian blur on the input
  // image. Runs BEFORE AffineWarp; writes blurred_input_image which AffineWarp
  // then samples. σ=0 yields a pass-through copy (identity path).
  // Phase C-1: per-slot — slot s binds mem->slots[s].input_image_view (in) +
  // mem->slots[s].blurred_input_image_view (out).
  VkDescriptorSetLayout preblur_desc_set_layout;
  VkDescriptorPool preblur_desc_pool;
  VkDescriptorSet preblur_desc_set[VKSIFT_MAX_PYRAMID_SLOTS];
  VkPipelineLayout preblur_pipeline_layout;
  VkPipeline preblur_pipeline;

  // PreBlur params consumed by recScaleSpaceConstructionCmds. Caller-settable
  // via vksift_setPendingPreBlur(). Default in createSiftDetector is σ=0
  // (pass-through identity blur).
  float pending_blur_sigma;
  float pending_blur_dir_x, pending_blur_dir_y;
  bool  pending_blur_dirty;

  // AffineWarp set — used by the ASIFT batch detect path to pre-warp the
  // input image before each pyramid build. Idle on the standard detect path.
  // Phase C-1: per-slot — slot s binds mem->slots[s].blurred_input_image_view
  // (in) + mem->slots[s].warped_input_image_view (out).
  VkDescriptorSetLayout affinewarp_desc_set_layout;
  VkDescriptorPool affinewarp_desc_pool;
  VkDescriptorSet affinewarp_desc_set[VKSIFT_MAX_PYRAMID_SLOTS];
  VkPipelineLayout affinewarp_pipeline_layout;
  VkPipeline affinewarp_pipeline;

  // Affine matrix consumed by recScaleSpaceConstructionCmds for the AffineWarp
  // dispatch on octave 0. Caller-settable via vksift_setPendingAffineWarp().
  // Default value (set in createSiftDetector) is the identity matrix.
  float pending_warp_a11, pending_warp_a12, pending_warp_a13;
  float pending_warp_a21, pending_warp_a22, pending_warp_a23;
  float pending_warp_fill;
  bool  pending_warp_dirty;

  // Gaussian Blur set
  // Phase C-1: blur_desc_sets is a flat array of nb_pyramid_slots * max_nb_octaves * 2
  // sets, indexed as `blur_desc_sets[slot * (max_nb_octaves * 2) + pass * max_nb_octaves + oct]`
  // (pass=0 horizontal, pass=1 vertical). The blur_h_desc_sets / blur_v_desc_sets
  // arrays are flat per-slot views of size nb_pyramid_slots * max_nb_octaves
  // each — use the BLUR_DESC_SET helper macros in sift_detector.c to index.
  VkDescriptorSetLayout blur_desc_set_layout;
  VkDescriptorPool blur_desc_pool;
  VkDescriptorSet *blur_desc_sets;     // [N*max_oct*2]
  VkDescriptorSet *blur_h_desc_sets;   // [N*max_oct], = blur_desc_sets
  VkDescriptorSet *blur_v_desc_sets;   // [N*max_oct], = blur_desc_sets + N*max_oct
  VkPipelineLayout blur_pipeline_layout;
  VkPipeline blur_pipeline;
  // Difference of Gaussian set — flat [slot*max_nb_octaves + oct].
  VkDescriptorSetLayout dog_desc_set_layout;
  VkDescriptorPool dog_desc_pool;
  VkDescriptorSet *dog_desc_sets;
  VkPipelineLayout dog_pipeline_layout;
  VkPipeline dog_pipeline;
  // Downsample2x set (Lowe pixel-aligned octave downsample, replaces blit) —
  // flat [slot*max_nb_octaves + oct].
  VkDescriptorSetLayout downsample_desc_set_layout;
  VkDescriptorPool downsample_desc_pool;
  VkDescriptorSet *downsample_desc_sets;
  VkPipelineLayout downsample_pipeline_layout;
  VkPipeline downsample_pipeline;
  // QuantizeF32ToInput set — device-side R32F (slots[s].rotated_image, IMAS
  // output) → R8_UNORM (slots[s].input_image) copy with quantization. Replaces
  // the host roundtrip on the on-IMAS detect path.
  // Phase C-1: per-slot.
  VkDescriptorSetLayout quantize_desc_set_layout;
  VkDescriptorPool quantize_desc_pool;
  VkDescriptorSet quantize_desc_set[VKSIFT_MAX_PYRAMID_SLOTS];
  VkPipelineLayout quantize_pipeline_layout;
  VkPipeline quantize_pipeline;
  // Valid sub-region of mem->rotated_image the IMAS pipeline wrote into
  // (= run_imas's out_w / out_h). Set per-call before submitting
  // detection_command_buffer_from_imas. The quantize shader fills the rest
  // of input_image with quantize_fill_value (matches the host-roundtrip
  // path's pad_tilted layout).
  uint32_t quantize_valid_w;
  uint32_t quantize_valid_h;
  float    quantize_fill_value;
  // ExtractKeypoints set — flat [slot*max_nb_octaves + oct]. Slot s's
  // descriptor set binds sift_buffer_arr[s] so concurrent waves write to
  // independent feature buffers. Requires nb_sift_buffer >= nb_pyramid_slots
  // (clamped at create time — see prepareDescriptorSets).
  VkDescriptorSetLayout extractkpts_desc_set_layout;
  VkDescriptorPool extractkpts_desc_pool;
  VkDescriptorSet *extractkpts_desc_sets;
  VkPipelineLayout extractkpts_pipeline_layout;
  VkPipeline extractkpts_pipeline;
  VkPipeline extractkpts_2d_pipeline;
  // ComputeOrientation set — flat [slot*max_nb_octaves + oct].
  VkDescriptorSetLayout orientation_desc_set_layout;
  VkDescriptorPool orientation_desc_pool;
  VkDescriptorSet *orientation_desc_sets;
  VkPipelineLayout orientation_pipeline_layout;
  VkPipeline orientation_pipeline;
  // ComputeDescriptor set — flat [slot*max_nb_octaves + oct].
  VkDescriptorSetLayout descriptor_desc_set_layout;
  VkDescriptorPool descriptor_desc_pool;
  VkDescriptorSet *descriptor_desc_sets;
  VkPipelineLayout descriptor_pipeline_layout;
  VkPipeline descriptor_pipeline;

  // Phase D — BackProjectFeatures compute pipeline. Reads each feature in a
  // slot's sift_buffer octave section, applies the K·σ·σ_max parallelogram
  // boundary check, back-projects (x, y) from tilted-frame → input-frame.
  // Rejected features get octave_idx = -1 (Julia driver filters on read).
  //
  // Binding model: set = 0 is the per-(slot, octave) SIFT_buffer section
  // (descriptor created with `.offset = octave_section_offset_arr[oct]` so
  // the shader sees the section header at `nb_elem` + features at data[0..]).
  // set = 1 is the slot's WarpParamsUBO (read-only). Dispatch is one
  // workgroup per (slot, octave) with local_size_x = 64 stride-looping over
  // the section.
  //
  // backproject_desc_sets is a flat [slot*max_nb_octaves + oct] array indexed
  // via slot_oct_idx(detector, slot, oct), matching extractkpts_desc_sets.
  VkDescriptorSetLayout backproject_desc_set_layout;
  VkDescriptorPool backproject_desc_pool;
  VkDescriptorSet *backproject_desc_sets;
  VkPipelineLayout backproject_pipeline_layout;
  VkPipeline backproject_pipeline;

  // RGBA→Gray conversion set (only when use_rgba_input=true). Per-slot —
  // slot s binds mem->slots[s].rgba_input_image_view (in) +
  // mem->slots[s].input_image_view (out).
  VkDescriptorSetLayout rgba_convert_desc_set_layout;
  VkDescriptorPool rgba_convert_desc_pool;
  VkDescriptorSet rgba_convert_desc_set[VKSIFT_MAX_PYRAMID_SLOTS];
  VkPipelineLayout rgba_convert_pipeline_layout;
  VkPipeline rgba_convert_pipeline;

  // RGB→Gray conversion set (only when use_rgb_input=true). Per-slot —
  // slot s binds mem->slots[s].rgb_input_buffer (in) +
  // mem->slots[s].input_image_view (out).
  // Uses SSBO for RGB input since VK_FORMAT_R8G8B8 has poor storage image support
  VkDescriptorSetLayout rgb_convert_desc_set_layout;
  VkDescriptorPool rgb_convert_desc_pool;
  VkDescriptorSet rgb_convert_desc_set[VKSIFT_MAX_PYRAMID_SLOTS];
  VkPipelineLayout rgb_convert_pipeline_layout;
  VkPipeline rgb_convert_pipeline;

  // Config
  bool use_hardware_interp_kernel;
  float input_blur_level;
  float seed_scale_sigma;
  float intensity_threshold;
  float edge_threshold;
  uint32_t max_nb_orientations;
  uint32_t use_vlfeat_format;
  bool detection_only;
  bool use_2d_nms;
  bool use_rgba_input;
  bool use_rgb_input;

} * vksift_SiftDetector;

bool vksift_createSiftDetector(vkenv_Device device, vksift_SiftMemory memory, vksift_SiftDetector *detector_ptr, const vksift_Config *config);
void vksift_destroySiftDetector(vksift_SiftDetector *detector_ptr);

bool vksift_dispatchSiftDetection(vksift_SiftDetector detector, const uint32_t target_buffer_idx, const bool memory_layout_updated);

// Same as vksift_dispatchSiftDetection but submits the on-IMAS detection
// command buffer (which sources its input from mem->rotated_image instead of
// the staging buffer). Caller must have populated rotated_image with valid
// IMAS-tilted Float32 content and set detector->quantize_width / _height to
// the actual sub-region the IMAS pipeline wrote into.
bool vksift_dispatchSiftDetectionFromImas(vksift_SiftDetector detector, const uint32_t target_buffer_idx, const bool memory_layout_updated);

// Phase B-3: dispatch the fully-fused IMAS-chain + Quantize + SIFT-detect
// command buffer for the given slot. The host first populates the slot's
// WarpParamsUBO (affine matrix, sigma_aa, canvas dims, valid sub-region, ...)
// and the slot's indirect-dispatch buffer (group counts for AffineWarp,
// GaussBlur1D, Finvspline{Row,Col}, Fproj, Quantize), then submits the
// pre-recorded fused command buffer on detector->general_queue with
// detector->end_of_detection_fence. Detection results land in sift_buffer_arr[
// target_buffer_idx] and can be read via the existing
// vksift_getFeaturesNumber / vksift_downloadFeatures path.
//
// (W, H) are the input image dims (= curr_input_image_*; same for every warp
// in a run). (t_factor, theta_rad) define the IMAS warp. (canvas_w, canvas_h)
// is the SIFT pyramid input canvas (keep stable across warps).
//
// Phase C-1: every detector + IMAS descriptor set is now allocated per slot,
// and the fused cmd buffer recording for slot s binds slot s's image views +
// slot s's UBO. Concurrent submission of multiple slots' fused cmd buffers is
// safe from a descriptor-state standpoint; the sift_buffer binding uses
// sift_buffer_arr[s] so feature outputs are independent.
bool vksift_dispatchFusedImasWarpForSlot(vksift_SiftDetector detector,
                                         uint32_t slot_idx, const uint32_t target_buffer_idx,
                                         uint32_t W, uint32_t H,
                                         float t_factor, float theta_rad,
                                         uint32_t canvas_w, uint32_t canvas_h,
                                         bool memory_layout_updated);

// Phase C-3 helper: fill slot's WarpParamsUBO + SlotDispatchBuffer with the
// per-warp host-side params. No GPU submission; the caller is responsible for
// submitting (or batch-submitting) the slot's fused_imas_detect_command_buffer
// afterwards. warp_idx is the IMAS schedule index stamped into the UBO's
// warp_idx field (read by the back-projection shader in Phase D); for the
// serial entry point it's currently set to slot_idx.
void vksift_fillFusedWarpState(vksift_SiftDetector detector,
                               uint32_t slot_idx, uint32_t warp_idx,
                               uint32_t W, uint32_t H,
                               float t_factor, float theta_rad,
                               uint32_t canvas_w, uint32_t canvas_h);

// Phase C-3 helper: ensure the detector's command buffers (including the
// per-slot fused IMAS+detect ones) are recorded for the supplied
// target_buffer_idx and the current memory layout. Mirrors the gating logic
// from dispatchDetectionCmdBuffer / vksift_dispatchFusedImasWarpForSlot so the
// parallel-wave dispatcher can re-record once per wave when needed (memory
// layout changed, target_buffer_idx changed, blur dirty, etc.).
bool vksift_ensureDetectorCmdBuffersRecorded(vksift_SiftDetector detector,
                                             uint32_t target_buffer_idx,
                                             bool memory_layout_updated);

// Set the affine matrix that will be pushed to AffineWarp.comp on the next
// detect dispatch. Marks the command buffer for re-record. Matrix layout:
//   a_ij are entries of the 2x3 inverse affine A_inv mapping warped pixel
//   (col_out, row_out) → input pixel (col_in, row_in):
//       col_in = a11*col_out + a12*row_out + a13
//       row_in = a21*col_out + a22*row_out + a23
// fill_value is returned for out-of-bounds samples (in [0..1] normalized intensity).
// Set the σ_aa pre-blur applied to the input image before AffineWarp on the
// next detect dispatch. Direction (dir_x, dir_y) is the 1D blur axis in
// input pixel coords (ASIFT uses sin φ, cos φ for the squash direction).
// σ = 0 yields a pass-through copy (no blur, identity-equivalent).
// Marks the command buffer for re-record.
void vksift_setPendingPreBlur(vksift_SiftDetector detector,
                              float sigma, float dir_x, float dir_y);

void vksift_setPendingAffineWarp(vksift_SiftDetector detector,
                                 float a11, float a12, float a13,
                                 float a21, float a22, float a23,
                                 float fill_value);

#endif // VKSIFT_SIFTDETECTOR