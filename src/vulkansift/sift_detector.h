#ifndef VKSIFT_SIFTDETECTOR
#define VKSIFT_SIFTDETECTOR

#include "sift_memory.h"
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

  // Gaussian kernels
  uint32_t *gaussian_kernel_sizes;
  float *gaussian_kernels;

  // AffineWarp set — used by the ASIFT batch detect path to pre-warp the
  // input image before each pyramid build. Idle on the standard detect path.
  VkDescriptorSetLayout affinewarp_desc_set_layout;
  VkDescriptorPool affinewarp_desc_pool;
  VkDescriptorSet affinewarp_desc_set;
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
  VkDescriptorSetLayout blur_desc_set_layout;
  VkDescriptorPool blur_desc_pool;
  VkDescriptorSet *blur_desc_sets;
  VkDescriptorSet *blur_h_desc_sets;
  VkDescriptorSet *blur_v_desc_sets;
  VkPipelineLayout blur_pipeline_layout;
  VkPipeline blur_pipeline;
  // Difference of Gaussian set
  VkDescriptorSetLayout dog_desc_set_layout;
  VkDescriptorPool dog_desc_pool;
  VkDescriptorSet *dog_desc_sets;
  VkPipelineLayout dog_pipeline_layout;
  VkPipeline dog_pipeline;
  // ExtractKeypoints set
  VkDescriptorSetLayout extractkpts_desc_set_layout;
  VkDescriptorPool extractkpts_desc_pool;
  VkDescriptorSet *extractkpts_desc_sets;
  VkPipelineLayout extractkpts_pipeline_layout;
  VkPipeline extractkpts_pipeline;
  VkPipeline extractkpts_2d_pipeline;
  // ComputeOrientation set
  VkDescriptorSetLayout orientation_desc_set_layout;
  VkDescriptorPool orientation_desc_pool;
  VkDescriptorSet *orientation_desc_sets;
  VkPipelineLayout orientation_pipeline_layout;
  VkPipeline orientation_pipeline;
  // ComputeDescriptor set
  VkDescriptorSetLayout descriptor_desc_set_layout;
  VkDescriptorPool descriptor_desc_pool;
  VkDescriptorSet *descriptor_desc_sets;
  VkPipelineLayout descriptor_pipeline_layout;
  VkPipeline descriptor_pipeline;

  // RGBA→Gray conversion set (only when use_rgba_input=true)
  VkDescriptorSetLayout rgba_convert_desc_set_layout;
  VkDescriptorPool rgba_convert_desc_pool;
  VkDescriptorSet rgba_convert_desc_set;
  VkPipelineLayout rgba_convert_pipeline_layout;
  VkPipeline rgba_convert_pipeline;

  // RGB→Gray conversion set (only when use_rgb_input=true)
  // Uses SSBO for RGB input since VK_FORMAT_R8G8B8 has poor storage image support
  VkDescriptorSetLayout rgb_convert_desc_set_layout;
  VkDescriptorPool rgb_convert_desc_pool;
  VkDescriptorSet rgb_convert_desc_set;
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

// Set the affine matrix that will be pushed to AffineWarp.comp on the next
// detect dispatch. Marks the command buffer for re-record. Matrix layout:
//   a_ij are entries of the 2x3 inverse affine A_inv mapping warped pixel
//   (col_out, row_out) → input pixel (col_in, row_in):
//       col_in = a11*col_out + a12*row_out + a13
//       row_in = a21*col_out + a22*row_out + a23
// fill_value is returned for out-of-bounds samples (in [0..1] normalized intensity).
void vksift_setPendingAffineWarp(vksift_SiftDetector detector,
                                 float a11, float a12, float a13,
                                 float a21, float a22, float a23,
                                 float fill_value);

#endif // VKSIFT_SIFTDETECTOR