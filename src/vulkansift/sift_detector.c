#include "sift_detector.h"
#include "sift_imas.h"

#include "vkenv/logger.h"
#include "vkenv/vulkan_utils.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

static const char LOG_TAG[] = "SiftDetector";

typedef struct
{
  uint32_t is_vertical;
  uint32_t array_layer;
  uint32_t kernel_size;
  float kernel[VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE];
} GaussianBlurPushConsts;

typedef struct
{
  int32_t octave_idx;
  float seed_scale_sigma;
  float dog_threshold;
  float edge_threshold;
  int32_t nb_scales;       // = nb_scales_per_octave; sigma divisor in shader,
                           // independent of dispatch range so boundary-aware
                           // 3D NMS (dispatch = nb_scales + 2) keeps σ correct.
  int32_t use_upsampling;  // 1 if first_octave=-1; shader compensates the
                           // +0.25 input-pixel bias from the VK_FILTER_LINEAR
                           // upsample blit (which propagates through every
                           // Downsample2x).
} ExtractKeypointsPushConsts;

// Push constants for PreBlur1D.comp: sigma + direction vector (3 × float = 12 B).
// PreBlur1D stays on push constants in Phase B-2 (plan explicitly leaves it
// as optional). AffineWarp + QuantizeF32ToInput migrated to the WarpParamsUBO
// (set = 1, binding = 0) — see sift_warp_ubo.h.
typedef struct
{
  float sigma;
  float dir_x;
  float dir_y;
} PreBlur1DPushConsts;

// Push constants for Downsample2x.comp: layer indices + dst dims (16 B).
typedef struct
{
  int32_t src_layer;
  int32_t dst_layer;
  int32_t dst_width;
  int32_t dst_height;
} Downsample2xPushConsts;

// Push constants for Upsample2xLinear.comp: dst layer + dst/src dims (20 B).
typedef struct
{
  int32_t dst_layer;
  int32_t dst_width;
  int32_t dst_height;
  int32_t src_width;
  int32_t src_height;
} Upsample2xLinearPushConsts;

// Shader-region timestamp boundaries written into the per-slot query pool by
// recFusedImasDetectCmdsForSlot. Each is a vkCmdWriteTimestamp at the END of
// the named region (region [i] = TS[i+1] - TS[i]).
enum {
  VKSIFT_TS_START         = 0,
  VKSIFT_TS_AFTER_IMAS    = 1,  // rotate + finvspline + fproj + warp
  VKSIFT_TS_AFTER_QUANT   = 2,  // QuantizeF32ToInput
  VKSIFT_TS_AFTER_SCALES  = 3,  // ClearBuffer + ScaleSpace × all octaves
  VKSIFT_TS_AFTER_DOG     = 4,  // DifferenceOfGaussian × all octaves
  VKSIFT_TS_AFTER_EXTRACT = 5,  // ExtractKeypoints × all octaves
  VKSIFT_TS_AFTER_ORIDESC = 6,  // Orientation + Descriptor (if !detection_only)
  VKSIFT_TS_END           = 7,  // CopySIFTCount + BackProjectFeatures
  VKSIFT_NUM_TS           = 8
};

static void getGPUDebugMarkerFuncs(vksift_SiftDetector detector)
{
  detector->vkCmdDebugMarkerBeginEXT = (PFN_vkCmdDebugMarkerBeginEXT)vkGetDeviceProcAddr(detector->dev->device, "vkCmdDebugMarkerBeginEXT");
  detector->vkCmdDebugMarkerEndEXT = (PFN_vkCmdDebugMarkerEndEXT)vkGetDeviceProcAddr(detector->dev->device, "vkCmdDebugMarkerEndEXT");
  detector->debug_marker_supported = (detector->vkCmdDebugMarkerBeginEXT != NULL) && (detector->vkCmdDebugMarkerEndEXT != NULL);
}

static void beginMarkerRegion(vksift_SiftDetector detector, VkCommandBuffer cmd_buf, const char *region_name)
{
  if (detector->debug_marker_supported)
  {
    VkDebugMarkerMarkerInfoEXT marker_info = {.sType = VK_STRUCTURE_TYPE_DEBUG_MARKER_MARKER_INFO_EXT, .pMarkerName = region_name};
    detector->vkCmdDebugMarkerBeginEXT(cmd_buf, &marker_info);
  }
}
static void endMarkerRegion(vksift_SiftDetector detector, VkCommandBuffer cmd_buf)
{
  if (detector->debug_marker_supported)
  {
    detector->vkCmdDebugMarkerEndEXT(cmd_buf);
  }
}

static void setupGaussianKernels(vksift_SiftDetector detector)
{
  uint32_t nb_scales = detector->mem->nb_scales_per_octave;
  detector->gaussian_kernels = (float *)malloc(sizeof(float) * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE * (nb_scales + 3));
  detector->gaussian_kernel_sizes = (uint32_t *)malloc(sizeof(uint32_t) * (nb_scales + 3));
  for (uint32_t i = 0; i < VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE * (nb_scales + 3); i++)
  {
    detector->gaussian_kernels[i] = 0.f;
  }
  memset(detector->gaussian_kernel_sizes, 0, sizeof(uint32_t) * (nb_scales + 3));

  for (uint32_t scale_i = 0; scale_i < (nb_scales + 3); scale_i++)
  {
    // The idea of the kernels in the scale space is that each scale is a blurred version of the previous scale.
    // The first scale of the pyramid (seed scale means scale 0 at octave 0), has a blur level defied by the configuration
    // (default to 1.6 from Lowe's paper). We take into account the initial blur effect of the input image (default 0.5) so the first kernel
    // perform a blurring operation to tranform the input image with defined blur level (0.5) to a seed image with defined blur level (1.6)
    // that will be used to sequentially build the scale space.
    // When building the scale-space, the idea if that each octave is 2x more blurred than the previous octave and every first scale of an octave
    // if a 2x downscaled version of the scale nb_scale (3) of the previous octave. Since the downscaling does not change the blur level, this means that
    // on the previous octave the scale nb_scale (3) is 2x more blurred than the scale 0 of the same octave. This sets the constraint that each
    // scale is 2^(1.0/nb_scale)*seed_scale_sigma more blurred that the previous one, since for the scale nb_scale 2^(nb_scale/nb_scale) = 2

    float sep_kernel_sigma;
    if (scale_i == 0)
    {
      // Used only for first octave (since all other first scales used downsampled scale from octave-1)
      // The seed scale initial blur level is doubled when 2x upsampling is used (basically account for the blur effect caused by the upsampling)
      float first_scale_init_blur_level = detector->mem->use_upsampling ? detector->input_blur_level * 2.f : detector->input_blur_level;
      sep_kernel_sigma = sqrtf((detector->seed_scale_sigma * detector->seed_scale_sigma) - (first_scale_init_blur_level * first_scale_init_blur_level));
    }
    else
    {
      // Gaussian blur from one scale to the other
      float sig_prev = powf(powf(2.f, 1.f / nb_scales), (float)(scale_i - 1)) * detector->seed_scale_sigma;
      float sig_total = sig_prev * powf(2.f, 1.f / nb_scales);
      sep_kernel_sigma = sqrtf(sig_total * sig_total - sig_prev * sig_prev);
    }

    // Compute the gaussian kernel for the defined sigma value
    uint32_t kernel_size = (int)(ceilf(sep_kernel_sigma * 4.f) + 1.f);
    if (kernel_size > VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE)
    {
      logWarning(LOG_TAG,
                 "Required Gaussian kernel size (%d) is higher that the max kernel size (%d), additional coefficients will be ignored. "
                 "Consider setting a smaller seed_scale_sigma value in the config structure.",
                 kernel_size, VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE);
      kernel_size = VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE;
    }
    detector->gaussian_kernel_sizes[scale_i] = kernel_size;

    float kernel_tmp_data[VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE];
    kernel_tmp_data[0] = 1.f;
    float sum_kernel = kernel_tmp_data[0];
    for (uint32_t i = 1; i < kernel_size; i++)
    {
      kernel_tmp_data[i] = exp(-0.5 * powf((float)(i), 2.f) / powf(sep_kernel_sigma, 2.f));
      sum_kernel += 2 * kernel_tmp_data[i];
    }

    logDebug(LOG_TAG, "Gaussian kernels");
    logDebug(LOG_TAG, "Scale %d sigma=%f kernel size=%d", scale_i, sep_kernel_sigma, kernel_size);
    for (uint32_t i = 0; i < kernel_size; i++)
    {
      kernel_tmp_data[i] /= sum_kernel;
      logDebug(LOG_TAG, "%f", kernel_tmp_data[i]);
    }

    float *scale_kernel = &detector->gaussian_kernels[scale_i * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE];
    // Compute hardware interpolated kernel if necessary
    if (detector->use_hardware_interp_kernel)
    {
      // Hardware interpolated kernel based on https://rastergrid.com/blog/2010/09/efficient-gaussian-blur-with-linear-sampling/
      // The goal here is to use hardware sampler to reduce the number of texture fetch by a factor of two
      // The same space is used in the kernel buffer, number of coeff is /2 but we add the texture offset to the buffer
      // First kernel coeff and offset stays the same
      scale_kernel[0] = kernel_tmp_data[0];
      scale_kernel[1] = (float)0;
      for (uint32_t data_idx = 1, kern_idx = 1; (data_idx + 1) < kernel_size; data_idx += 2, kern_idx++)
      {
        scale_kernel[kern_idx * 2] = kernel_tmp_data[data_idx] + kernel_tmp_data[data_idx + 1];
        scale_kernel[(kern_idx * 2) + 1] = (((float)data_idx * kernel_tmp_data[data_idx]) + ((float)(data_idx + 1) * kernel_tmp_data[data_idx + 1])) /
                                           (kernel_tmp_data[data_idx] + kernel_tmp_data[data_idx + 1]);
      }
    }
    else
    {
      for (uint32_t i = 0; i < kernel_size; i++)
      {
        scale_kernel[i] = kernel_tmp_data[i];
      }
    }
  }
}

static bool setupCommandPools(vksift_SiftDetector detector)
{
  VkCommandPoolCreateInfo pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                       .queueFamilyIndex = detector->dev->general_queues_family_idx};
  if (vkCreateCommandPool(detector->dev->device, &pool_info, NULL, &detector->general_command_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create the general-purpose command pool");
    return false;
  }

  if (detector->dev->async_transfer_available)
  {
    VkCommandPoolCreateInfo async_pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                               .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                               .queueFamilyIndex = detector->dev->async_transfer_queues_family_idx};
    if (vkCreateCommandPool(detector->dev->device, &async_pool_info, NULL, &detector->async_transfer_command_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create the asynchronous transfer command pool");
      return false;
    }
  }

  // Phase C-async: pool for cmd buffers submitted on the dedicated compute
  // queue (parallel-IMAS multi-queue dispatch). On devices without a separate
  // compute queue family this stays NULL and the dispatcher falls back to the
  // single-queue path.
  if (detector->dev->async_compute_available)
  {
    VkCommandPoolCreateInfo compute_pool_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                                 .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                                 .queueFamilyIndex = detector->dev->async_compute_queues_family_idx};
    if (vkCreateCommandPool(detector->dev->device, &compute_pool_info, NULL, &detector->async_compute_command_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create the asynchronous compute command pool");
      return false;
    }
  }

  return true;
}

static bool allocateCommandBuffers(vksift_SiftDetector detector)
{
  VkCommandBufferAllocateInfo allocate_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                               .pNext = NULL,
                                               .commandPool = detector->general_command_pool,
                                               .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                               .commandBufferCount = 1};
  if (vkAllocateCommandBuffers(detector->dev->device, &allocate_info, &detector->detection_command_buffer) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate the detection command buffe");
    return false;
  }
  if (vkAllocateCommandBuffers(detector->dev->device, &allocate_info, &detector->detection_command_buffer_from_imas) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate the detection-from-IMAS command buffer");
    return false;
  }
  // Phase B-3: per-slot fused IMAS-chain + Quantize + SIFT-detect command
  // buffers. Allocated up-front for every slot so the parallel-wave driver
  // (Phase C) can submit them concurrently without further allocation churn.
  for (uint32_t s = 0u; s < VKSIFT_MAX_PYRAMID_SLOTS; ++s)
  {
    if (vkAllocateCommandBuffers(detector->dev->device, &allocate_info,
                                 &detector->fused_imas_detect_command_buffer[s]) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to allocate the fused IMAS+detect command buffer (slot %u)", s);
      return false;
    }
  }

  // Phase C-async: mirror per-slot fused cmd buffers on the async-compute pool
  // so the parallel-IMAS dispatcher can submit half of every wave to
  // async_compute_queues[0]. Same recording content (filled in by
  // recordCommandBuffers); we just need a cmd buffer bound to the right
  // queue family to submit on that queue.
  if (detector->dev->async_compute_available)
  {
    VkCommandBufferAllocateInfo compute_allocate_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                                         .pNext = NULL,
                                                         .commandPool = detector->async_compute_command_pool,
                                                         .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                         .commandBufferCount = 1};
    for (uint32_t s = 0u; s < VKSIFT_MAX_PYRAMID_SLOTS; ++s)
    {
      if (vkAllocateCommandBuffers(detector->dev->device, &compute_allocate_info,
                                   &detector->fused_imas_detect_command_buffer_compute[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate the fused IMAS+detect command buffer on async-compute pool (slot %u)", s);
        return false;
      }
    }
  }
  else
  {
    for (uint32_t s = 0u; s < VKSIFT_MAX_PYRAMID_SLOTS; ++s)
    {
      detector->fused_imas_detect_command_buffer_compute[s] = VK_NULL_HANDLE;
    }
  }

  // If the async tranfer queue is available the SIFT buffers are owned by the transfer queue family
  // in this case we need to release this ownership from the transfer queue before using the buffers in the general purpose queue
  if (detector->dev->async_transfer_available)
  {
    allocate_info.commandPool = detector->async_transfer_command_pool;
    if (vkAllocateCommandBuffers(detector->dev->device, &allocate_info, &detector->release_buffer_ownership_command_buffer) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to allocate the release-buffer-ownership command buffer on the async transfer pool");
      return false;
    }
    if (vkAllocateCommandBuffers(detector->dev->device, &allocate_info, &detector->acquire_buffer_ownership_command_buffer) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to allocate the acquire-buffer-ownership command buffer on the async transfer pool");
      return false;
    }
  }

  return true;
}

static bool setupImageSampler(vksift_SiftDetector detector)
{
  VkSamplerCreateInfo sampler_info = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                      .pNext = NULL,
                                      .flags = 0,
                                      .magFilter = VK_FILTER_LINEAR,
                                      .minFilter = VK_FILTER_LINEAR,
                                      .mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR,
                                      .addressModeU = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                      .addressModeV = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                      .addressModeW = VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
                                      .mipLodBias = 0.f,
                                      .anisotropyEnable = VK_FALSE,
                                      .maxAnisotropy = 0.f,
                                      .compareEnable = VK_FALSE,
                                      .compareOp = VK_COMPARE_OP_ALWAYS,
                                      .minLod = 0.f,
                                      .maxLod = 0.f,
                                      .borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK,
                                      .unnormalizedCoordinates = VK_FALSE};

  if (vkCreateSampler(detector->dev->device, &sampler_info, NULL, &detector->image_sampler) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create image sampler");
    return false;
  }
  return true;
}

static VkDescriptorSetLayout *allocMultLayoutCopy(VkDescriptorSetLayout layout, int nb_copy)
{
  VkDescriptorSetLayout *layout_arr = (VkDescriptorSetLayout *)malloc(sizeof(VkDescriptorSetLayout) * nb_copy);
  for (int i = 0; i < nb_copy; i++)
  {
    layout_arr[i] = layout;
  }
  return layout_arr;
}

static bool prepareDescriptorSets(vksift_SiftDetector detector)
{
  VkResult alloc_res;
  // Phase C-1: pool sizing factor for per-slot descriptor sets. Most per-slot
  // pools are sized at exactly nb_pyramid_slots (active slots only — descriptor
  // sets for slot indices >= nb_pyramid_slots are never allocated because the
  // underlying images don't exist).
  uint32_t N = detector->mem->nb_pyramid_slots;
  if (N == 0u) N = 1u;
  const uint32_t max_oct = detector->mem->max_nb_octaves;
  ///////////////////////////////////////////////////
  // Resource bindings for PreBlur1D pipeline (ASIFT σ_aa pre-blur)
  // binding 0 = sampler2D (input_image), binding 1 = image2D r32f (blurred_input_image)
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding pb_in = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding pb_out = {.binding = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding pb_bindings[2] = {pb_in, pb_out};
    VkDescriptorSetLayoutCreateInfo pb_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = pb_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &pb_layout_info, NULL, &detector->preblur_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create PreBlur1D descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize pb_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    VkDescriptorPoolCreateInfo pb_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = N, .poolSizeCount = 2, .pPoolSizes = pb_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &pb_pool_info, NULL, &detector->preblur_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create PreBlur1D descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo pb_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = detector->preblur_desc_pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &detector->preblur_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &pb_alloc_info, &detector->preblur_desc_set[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate PreBlur1D descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Shared WarpParamsUBO descriptor set (set = 1 in AffineWarp.comp +
  // QuantizeF32ToInput.comp). One layout, one pool sized for nb_pyramid_slots
  // sets — each set's binding 0 references mem->slots[s].warp_params_ubo.
  // Phase B-2 only binds warp_ubo_desc_sets[0]; the slot[1..] descriptors
  // are wired up for forward-compatibility with the Phase C parallel waves.
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding ubo_binding = {.binding = 0,
                                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                                .descriptorCount = 1,
                                                .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutCreateInfo ubo_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &ubo_binding};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &ubo_layout_info, NULL,
                                    &detector->warp_ubo_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create WarpParamsUBO descriptor set layout");
      return false;
    }
    uint32_t n_slots = detector->mem->nb_pyramid_slots;
    if (n_slots == 0u) n_slots = 1u;
    VkDescriptorPoolSize ubo_pool_size = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = n_slots};
    VkDescriptorPoolCreateInfo ubo_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .maxSets = n_slots, .poolSizeCount = 1,
                                                .pPoolSizes = &ubo_pool_size};
    if (vkCreateDescriptorPool(detector->dev->device, &ubo_pool_info, NULL,
                               &detector->warp_ubo_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create WarpParamsUBO descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < n_slots; ++s)
    {
      VkDescriptorSetAllocateInfo ubo_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                    .descriptorPool = detector->warp_ubo_desc_pool,
                                                    .descriptorSetCount = 1,
                                                    .pSetLayouts = &detector->warp_ubo_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &ubo_alloc_info,
                                   &detector->warp_ubo_desc_sets[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate WarpParamsUBO descriptor set %u", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Resource bindings for AffineWarp pipeline (ASIFT batch path)
  // binding 0 = sampler2D (blurred_input_image), binding 1 = image2D r32f (warped_input_image)
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding aw_in = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding aw_out = {.binding = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding aw_bindings[2] = {aw_in, aw_out};
    VkDescriptorSetLayoutCreateInfo aw_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = aw_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &aw_layout_info, NULL, &detector->affinewarp_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create AffineWarp descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize aw_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    VkDescriptorPoolCreateInfo aw_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = N, .poolSizeCount = 2, .pPoolSizes = aw_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &aw_pool_info, NULL, &detector->affinewarp_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create AffineWarp descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo aw_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = detector->affinewarp_desc_pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &detector->affinewarp_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &aw_alloc_info, &detector->affinewarp_desc_set[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate AffineWarp descriptor set (slot %u)", s);
        return false;
      }
    }
    // Image-view bindings (input_image as sampler source, warped_input_image as
    // storage output) are wired up by writeDescriptorSets in the
    // detect-dispatch path so they track allocation-on-resize.
  }

  ///////////////////////////////////////////////////
  // Descriptors for GaussianBlur pipeline
  ///////////////////////////////////////////////////
  VkDescriptorSetLayoutBinding blur_input_layout_binding = {.binding = 0,
                                                            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                            .descriptorCount = 1,
                                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                            .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding blur_output_layout_binding = {.binding = 1,
                                                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                             .descriptorCount = 1,
                                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                             .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding blur_bindings[2] = {blur_input_layout_binding, blur_output_layout_binding};
  VkDescriptorSetLayoutCreateInfo blur_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = blur_bindings};
  if (vkCreateDescriptorSetLayout(detector->dev->device, &blur_layout_info, NULL, &detector->blur_desc_set_layout) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create GaussianBlur descriptor set layout");
    return false;
  }

  // Create descriptor pool to allocate descriptor sets (reserve x2 for the
  // horizontal and vertical pass, x N for per-slot bindings)
  VkDescriptorPoolSize blur_pool_sizes[2];
  blur_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = max_oct * 2u * N};
  blur_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * 2u * N};
  VkDescriptorPoolCreateInfo blur_descriptor_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                          .maxSets = max_oct * 2u * N,
                                                          .poolSizeCount = 2,
                                                          .pPoolSizes = blur_pool_sizes};
  if (vkCreateDescriptorPool(detector->dev->device, &blur_descriptor_pool_info, NULL, &detector->blur_desc_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create GaussianBlur descriptor pool");
    return false;
  }

  // Create descriptor sets that can be bound in command buffer. Layout:
  //   blur_desc_sets[0 .. N*max_oct)            : horizontal pass (h_desc_sets)
  //   blur_desc_sets[N*max_oct .. 2*N*max_oct)  : vertical pass   (v_desc_sets)
  // Within each half, slot s octave o lives at [s*max_oct + o].
  const uint32_t blur_total = max_oct * 2u * N;
  VkDescriptorSetLayout *blur_layouts = allocMultLayoutCopy(detector->blur_desc_set_layout, blur_total);
  detector->blur_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * blur_total);
  detector->blur_h_desc_sets = detector->blur_desc_sets;
  detector->blur_v_desc_sets = detector->blur_desc_sets + (max_oct * N);
  VkDescriptorSetAllocateInfo blur_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                 .descriptorPool = detector->blur_desc_pool,
                                                 .descriptorSetCount = blur_total,
                                                 .pSetLayouts = blur_layouts};

  alloc_res = vkAllocateDescriptorSets(detector->dev->device, &blur_alloc_info, detector->blur_desc_sets);
  free(blur_layouts);
  if (alloc_res != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate GaussianBlur descriptor set");
    return false;
  }

  ///////////////////////////////////////////////////
  // Descriptors for DifferenceOfGaussian pipeline
  ///////////////////////////////////////////////////
  VkDescriptorSetLayoutBinding dog_input_layout_binding = {.binding = 0,
                                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                           .descriptorCount = 1,
                                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                           .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding dog_output_layout_binding = {.binding = 1,
                                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                            .descriptorCount = 1,
                                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                            .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding dog_bindings[2] = {dog_input_layout_binding, dog_output_layout_binding};

  VkDescriptorSetLayoutCreateInfo dog_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = dog_bindings};

  if (vkCreateDescriptorSetLayout(detector->dev->device, &dog_layout_info, NULL, &detector->dog_desc_set_layout) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create DifferenceOfGaussian descriptor set layout");
    return false;
  }

  // Create descriptor pool to allocate descriptor sets (per-slot per-octave)
  VkDescriptorPoolSize dog_pool_sizes[2];
  dog_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N};
  dog_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N};
  VkDescriptorPoolCreateInfo dog_descriptor_pool_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = max_oct * N, .poolSizeCount = 2, .pPoolSizes = dog_pool_sizes};
  if (vkCreateDescriptorPool(detector->dev->device, &dog_descriptor_pool_info, NULL, &detector->dog_desc_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create DifferenceOfGaussian descriptor pool");
    return false;
  }

  // Create descriptor sets that can be bound in command buffer. Flat layout:
  // [slot * max_oct + oct].
  VkDescriptorSetLayout *dog_layouts = allocMultLayoutCopy(detector->dog_desc_set_layout, max_oct * N);
  detector->dog_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
  VkDescriptorSetAllocateInfo dog_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                .descriptorPool = detector->dog_desc_pool,
                                                .descriptorSetCount = max_oct * N,
                                                .pSetLayouts = dog_layouts};
  alloc_res = vkAllocateDescriptorSets(detector->dev->device, &dog_alloc_info, detector->dog_desc_sets);
  free(dog_layouts);
  if (alloc_res != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate DifferenceOfGaussian descriptor set");
    return false;
  }

  ///////////////////////////////////////////////////
  // Descriptors for Downsample2x pipeline (octave transitions). One descriptor
  // set per octave i in [0, max_nb_octaves-1]; transition i reads from
  // octave_image_view_arr[i] (layer nb_scales) and writes to
  // octave_image_view_arr[i+1] (layer 0). The last set goes unused.
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding ds_in = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding ds_out = {.binding = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding ds_bindings[2] = {ds_in, ds_out};
    VkDescriptorSetLayoutCreateInfo ds_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = ds_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &ds_layout_info, NULL, &detector->downsample_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create Downsample2x descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize ds_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N}};
    VkDescriptorPoolCreateInfo ds_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = max_oct * N, .poolSizeCount = 2, .pPoolSizes = ds_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &ds_pool_info, NULL, &detector->downsample_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create Downsample2x descriptor pool");
      return false;
    }
    VkDescriptorSetLayout *ds_layouts = allocMultLayoutCopy(detector->downsample_desc_set_layout, max_oct * N);
    detector->downsample_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
    VkDescriptorSetAllocateInfo ds_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                 .descriptorPool = detector->downsample_desc_pool,
                                                 .descriptorSetCount = max_oct * N,
                                                 .pSetLayouts = ds_layouts};
    alloc_res = vkAllocateDescriptorSets(detector->dev->device, &ds_alloc_info, detector->downsample_desc_sets);
    free(ds_layouts);
    if (alloc_res != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to allocate Downsample2x descriptor set");
      return false;
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for Upsample2xLinear pipeline (octave-0 2× linear upsample when
  // use_upsampling=true). One set per slot — binding 0 = warped_input_image
  // (image2D r32f), binding 1 = octave_image_view_arr[0] (image2DArray r32f).
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding us_in = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding us_out = {.binding = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding us_bindings[2] = {us_in, us_out};
    VkDescriptorSetLayoutCreateInfo us_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = us_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &us_layout_info, NULL, &detector->upsample_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create Upsample2xLinear descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize us_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    VkDescriptorPoolCreateInfo us_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = N, .poolSizeCount = 2, .pPoolSizes = us_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &us_pool_info, NULL, &detector->upsample_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create Upsample2xLinear descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo us_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = detector->upsample_desc_pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &detector->upsample_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &us_alloc_info, &detector->upsample_desc_set[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate Upsample2xLinear descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for SiftSeedFromInput pipeline (IMAS-fused fast path).
  // binding 0 = STORAGE_IMAGE r8 (input_image), binding 1 = STORAGE_IMAGE r32f
  // image2DArray (octave_image_view_arr[0]). One set per slot.
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding sf_in = {.binding = 0,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding sf_out = {.binding = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                           .descriptorCount = 1,
                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                           .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding sf_bindings[2] = {sf_in, sf_out};
    VkDescriptorSetLayoutCreateInfo sf_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = sf_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &sf_layout_info, NULL, &detector->seed_from_input_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create SiftSeedFromInput descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize sf_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    VkDescriptorPoolCreateInfo sf_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = N, .poolSizeCount = 2, .pPoolSizes = sf_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &sf_pool_info, NULL, &detector->seed_from_input_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create SiftSeedFromInput descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo sf_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                   .descriptorPool = detector->seed_from_input_desc_pool,
                                                   .descriptorSetCount = 1,
                                                   .pSetLayouts = &detector->seed_from_input_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &sf_alloc_info, &detector->seed_from_input_desc_set[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate SiftSeedFromInput descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for QuantizeF32ToInput pipeline. Single descriptor set,
  // bindings 0 = STORAGE_IMAGE r32f (rotated_image, IMAS output) and
  // 1 = STORAGE_IMAGE r8 (input_image). Used by the on-IMAS detect path
  // to bridge device-side R32F → R8 without a host roundtrip.
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding q_in = {.binding = 0,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                         .descriptorCount = 1,
                                         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                         .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding q_out = {.binding = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .descriptorCount = 1,
                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                          .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding q_bindings[2] = {q_in, q_out};
    VkDescriptorSetLayoutCreateInfo q_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = q_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &q_layout_info, NULL, &detector->quantize_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create QuantizeF32ToInput descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize q_pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    VkDescriptorPoolCreateInfo q_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = N, .poolSizeCount = 2, .pPoolSizes = q_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &q_pool_info, NULL, &detector->quantize_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create QuantizeF32ToInput descriptor pool");
      return false;
    }
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo q_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                  .descriptorPool = detector->quantize_desc_pool,
                                                  .descriptorSetCount = 1,
                                                  .pSetLayouts = &detector->quantize_desc_set_layout};
      if (vkAllocateDescriptorSets(detector->dev->device, &q_alloc_info, &detector->quantize_desc_set[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate QuantizeF32ToInput descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for ExtractKeypoints pipeline
  ///////////////////////////////////////////////////
  VkDescriptorSetLayoutBinding extkpts_dog_image_layout_binding = {.binding = 0,
                                                                   .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                   .descriptorCount = 1,
                                                                   .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                   .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding extkpts_sift_buffer_layout_binding = {.binding = 1,
                                                                     .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                     .descriptorCount = 1,
                                                                     .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                     .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding extkpts_indispatch_buffer_layout_binding = {.binding = 2,
                                                                           .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                           .descriptorCount = 1,
                                                                           .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                           .pImmutableSamplers = NULL};

  VkDescriptorSetLayoutBinding extkpts_bindings[3] = {extkpts_dog_image_layout_binding, extkpts_sift_buffer_layout_binding,
                                                      extkpts_indispatch_buffer_layout_binding};

  VkDescriptorSetLayoutCreateInfo extkpts_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = extkpts_bindings};

  if (vkCreateDescriptorSetLayout(detector->dev->device, &extkpts_layout_info, NULL, &detector->extractkpts_desc_set_layout) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints descriptor set layout");
    return false;
  }

  // Create descriptor pool to allocate descriptor sets (per-slot per-octave)
  VkDescriptorPoolSize extkpts_pool_sizes[3];
  extkpts_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N};
  extkpts_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N};
  extkpts_pool_sizes[2] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N};
  VkDescriptorPoolCreateInfo extkpts_descriptor_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                             .maxSets = max_oct * N,
                                                             .poolSizeCount = 3,
                                                             .pPoolSizes = extkpts_pool_sizes};
  if (vkCreateDescriptorPool(detector->dev->device, &extkpts_descriptor_pool_info, NULL, &detector->extractkpts_desc_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints descriptor pool");
    return false;
  }

  // Create descriptor sets that can be bound in command buffer. Flat layout
  // [slot * max_oct + oct].
  VkDescriptorSetLayout *extkpts_layouts = allocMultLayoutCopy(detector->extractkpts_desc_set_layout, max_oct * N);
  detector->extractkpts_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
  VkDescriptorSetAllocateInfo extkpts_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                    .descriptorPool = detector->extractkpts_desc_pool,
                                                    .descriptorSetCount = max_oct * N,
                                                    .pSetLayouts = extkpts_layouts};

  alloc_res = vkAllocateDescriptorSets(detector->dev->device, &extkpts_alloc_info, detector->extractkpts_desc_sets);
  free(extkpts_layouts);
  if (alloc_res != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate ExtractKeypoints descriptor set");
    return false;
  }

  ///////////////////////////////////////////////////
  // Phase D — BackProjectFeatures descriptor sets (per-slot per-octave).
  // Layout has a single binding (the SIFT_buffer section); the WarpParamsUBO
  // is bound via set = 1 using the shared detector->warp_ubo_desc_sets[slot].
  // Slot s's per-octave sets bind slot_sift_buffer_idx(s)'s sift_buffer with
  // the octave's offset/range baked into the descriptor write.
  ///////////////////////////////////////////////////
  {
    VkDescriptorSetLayoutBinding bp_sift_buffer_binding = {.binding = 0,
                                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                            .descriptorCount = 1,
                                                            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                            .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutCreateInfo bp_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &bp_sift_buffer_binding};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &bp_layout_info, NULL,
                                    &detector->backproject_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create BackProjectFeatures descriptor set layout");
      return false;
    }
    VkDescriptorPoolSize bp_pool_sizes[1] = {
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N}};
    VkDescriptorPoolCreateInfo bp_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                .maxSets = max_oct * N,
                                                .poolSizeCount = 1,
                                                .pPoolSizes = bp_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &bp_pool_info, NULL,
                               &detector->backproject_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create BackProjectFeatures descriptor pool");
      return false;
    }
    VkDescriptorSetLayout *bp_layouts = allocMultLayoutCopy(detector->backproject_desc_set_layout, max_oct * N);
    detector->backproject_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
    VkDescriptorSetAllocateInfo bp_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                  .descriptorPool = detector->backproject_desc_pool,
                                                  .descriptorSetCount = max_oct * N,
                                                  .pSetLayouts = bp_layouts};
    alloc_res = vkAllocateDescriptorSets(detector->dev->device, &bp_alloc_info, detector->backproject_desc_sets);
    free(bp_layouts);
    if (alloc_res != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to allocate BackProjectFeatures descriptor sets");
      return false;
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for ComputeOrientation pipeline
  ///////////////////////////////////////////////////
  VkDescriptorSetLayoutBinding orientation_octave_image_layout_binding = {.binding = 0,
                                                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                          .descriptorCount = 1,
                                                                          .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                          .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding orientation_sift_buffer_layout_binding = {.binding = 1,
                                                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                         .descriptorCount = 1,
                                                                         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                         .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding orientation_indispatch_buffer_layout_binding = {.binding = 2,
                                                                               .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                               .descriptorCount = 1,
                                                                               .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                               .pImmutableSamplers = NULL};

  VkDescriptorSetLayoutBinding orientation_bindings[3] = {orientation_octave_image_layout_binding, orientation_sift_buffer_layout_binding,
                                                          orientation_indispatch_buffer_layout_binding};
  VkDescriptorSetLayoutCreateInfo orientation_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 3, .pBindings = orientation_bindings};

  if (vkCreateDescriptorSetLayout(detector->dev->device, &orientation_layout_info, NULL, &detector->orientation_desc_set_layout) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ComputeOrientation descriptor set layout");
    return false;
  }

  // Create descriptor pool to allocate descriptor sets (per-slot per-octave)
  VkDescriptorPoolSize orientation_pool_sizes[3];
  orientation_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N};
  orientation_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N};
  orientation_pool_sizes[2] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N};
  VkDescriptorPoolCreateInfo orientation_descriptor_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                                 .maxSets = max_oct * N,
                                                                 .poolSizeCount = 3,
                                                                 .pPoolSizes = orientation_pool_sizes};
  if (vkCreateDescriptorPool(detector->dev->device, &orientation_descriptor_pool_info, NULL, &detector->orientation_desc_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ComputeOrientation descriptor pool");
    return false;
  }

  // Create descriptor sets that can be bound in command buffer. Flat layout
  // [slot * max_oct + oct].
  VkDescriptorSetLayout *orientation_layouts = allocMultLayoutCopy(detector->orientation_desc_set_layout, max_oct * N);
  detector->orientation_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
  VkDescriptorSetAllocateInfo orientation_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                        .descriptorPool = detector->orientation_desc_pool,
                                                        .descriptorSetCount = max_oct * N,
                                                        .pSetLayouts = orientation_layouts};

  alloc_res = vkAllocateDescriptorSets(detector->dev->device, &orientation_alloc_info, detector->orientation_desc_sets);
  free(orientation_layouts);
  if (alloc_res != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate ComputeOrientation descriptor set");
    return false;
  }

  ///////////////////////////////////////////////////
  // Descriptors for ComputeDescriptors pipeline
  ///////////////////////////////////////////////////

  VkDescriptorSetLayoutBinding descriptor_octave_image_layout_binding = {.binding = 0,
                                                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                                         .descriptorCount = 1,
                                                                         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                         .pImmutableSamplers = NULL};
  VkDescriptorSetLayoutBinding descriptor_sift_buffer_layout_binding = {.binding = 1,
                                                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                                        .descriptorCount = 1,
                                                                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                                        .pImmutableSamplers = NULL};

  VkDescriptorSetLayoutBinding descriptor_bindings[2] = {descriptor_octave_image_layout_binding, descriptor_sift_buffer_layout_binding};

  VkDescriptorSetLayoutCreateInfo descriptor_layout_info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = descriptor_bindings};

  if (vkCreateDescriptorSetLayout(detector->dev->device, &descriptor_layout_info, NULL, &detector->descriptor_desc_set_layout) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ComputeDescriptors descriptor set layout");
    return false;
  }

  // Create descriptor pool to allocate descriptor sets (per-slot per-octave)
  VkDescriptorPoolSize descriptor_pool_sizes[2];
  descriptor_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = max_oct * N};
  descriptor_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = max_oct * N};
  VkDescriptorPoolCreateInfo descriptor_descriptor_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                                .maxSets = max_oct * N,
                                                                .poolSizeCount = 2,
                                                                .pPoolSizes = descriptor_pool_sizes};
  if (vkCreateDescriptorPool(detector->dev->device, &descriptor_descriptor_pool_info, NULL, &detector->descriptor_desc_pool) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create ComputeDescriptors descriptor pool");
    return false;
  }

  // Create descriptor sets that can be bound in command buffer. Flat layout
  // [slot * max_oct + oct].
  VkDescriptorSetLayout *descriptor_layouts = allocMultLayoutCopy(detector->descriptor_desc_set_layout, max_oct * N);
  detector->descriptor_desc_sets = (VkDescriptorSet *)malloc(sizeof(VkDescriptorSet) * max_oct * N);
  VkDescriptorSetAllocateInfo descriptor_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                       .descriptorPool = detector->descriptor_desc_pool,
                                                       .descriptorSetCount = max_oct * N,
                                                       .pSetLayouts = descriptor_layouts};

  alloc_res = vkAllocateDescriptorSets(detector->dev->device, &descriptor_alloc_info, detector->descriptor_desc_sets);
  free(descriptor_layouts);
  if (alloc_res != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to allocate ComputeDescriptors descriptor set");
    return false;
  }

  ///////////////////////////////////////////////////
  // Descriptors for RGBAtoGray pipeline (only when use_rgba_input)
  ///////////////////////////////////////////////////
  if (detector->use_rgba_input)
  {
    VkDescriptorSetLayoutBinding rgba_input_binding = {.binding = 0,
                                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                       .descriptorCount = 1,
                                                       .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                       .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding gray_output_binding = {.binding = 1,
                                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                        .descriptorCount = 1,
                                                        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                        .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding rgba_bindings[2] = {rgba_input_binding, gray_output_binding};
    VkDescriptorSetLayoutCreateInfo rgba_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = rgba_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &rgba_layout_info, NULL, &detector->rgba_convert_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create RGBAtoGray descriptor set layout");
      return false;
    }

    VkDescriptorPoolSize rgba_pool_sizes[1];
    rgba_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 2u * N};
    VkDescriptorPoolCreateInfo rgba_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                  .maxSets = N,
                                                  .poolSizeCount = 1,
                                                  .pPoolSizes = rgba_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &rgba_pool_info, NULL, &detector->rgba_convert_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create RGBAtoGray descriptor pool");
      return false;
    }

    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo rgba_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                     .descriptorPool = detector->rgba_convert_desc_pool,
                                                     .descriptorSetCount = 1,
                                                     .pSetLayouts = &detector->rgba_convert_desc_set_layout};
      alloc_res = vkAllocateDescriptorSets(detector->dev->device, &rgba_alloc_info, &detector->rgba_convert_desc_set[s]);
      if (alloc_res != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate RGBAtoGray descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  ///////////////////////////////////////////////////
  // Descriptors for RGBtoGray pipeline (only when use_rgb_input)
  // Binding 0: SSBO (packed RGB bytes), Binding 1: storage image (R8 output)
  ///////////////////////////////////////////////////
  if (detector->use_rgb_input)
  {
    VkDescriptorSetLayoutBinding rgb_ssbo_binding = {.binding = 0,
                                                      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                      .descriptorCount = 1,
                                                      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                      .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding rgb_gray_output_binding = {.binding = 1,
                                                             .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                             .descriptorCount = 1,
                                                             .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
                                                             .pImmutableSamplers = NULL};
    VkDescriptorSetLayoutBinding rgb_bindings[2] = {rgb_ssbo_binding, rgb_gray_output_binding};
    VkDescriptorSetLayoutCreateInfo rgb_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = rgb_bindings};
    if (vkCreateDescriptorSetLayout(detector->dev->device, &rgb_layout_info, NULL, &detector->rgb_convert_desc_set_layout) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create RGBtoGray descriptor set layout");
      return false;
    }

    VkDescriptorPoolSize rgb_pool_sizes[2];
    rgb_pool_sizes[0] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = N};
    rgb_pool_sizes[1] = (VkDescriptorPoolSize){.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N};
    VkDescriptorPoolCreateInfo rgb_pool_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                                 .maxSets = N,
                                                 .poolSizeCount = 2,
                                                 .pPoolSizes = rgb_pool_sizes};
    if (vkCreateDescriptorPool(detector->dev->device, &rgb_pool_info, NULL, &detector->rgb_convert_desc_pool) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create RGBtoGray descriptor pool");
      return false;
    }

    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorSetAllocateInfo rgb_alloc_info = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                                    .descriptorPool = detector->rgb_convert_desc_pool,
                                                    .descriptorSetCount = 1,
                                                    .pSetLayouts = &detector->rgb_convert_desc_set_layout};
      alloc_res = vkAllocateDescriptorSets(detector->dev->device, &rgb_alloc_info, &detector->rgb_convert_desc_set[s]);
      if (alloc_res != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to allocate RGBtoGray descriptor set (slot %u)", s);
        return false;
      }
    }
  }

  return true;
}

static bool setupComputePipelines(vksift_SiftDetector detector)
{
  //////////////////////////////////////
  // Setup PreBlur1D pipeline (ASIFT σ_aa pre-blur)
  //////////////////////////////////////
  {
    VkShaderModule pb_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/PreBlur1D.comp.spv", &pb_shader_module))
    {
      logError(LOG_TAG, "Failed to create PreBlur1D shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, pb_shader_module, detector->preblur_desc_set_layout, sizeof(PreBlur1DPushConsts),
                                     &detector->preblur_pipeline_layout, &detector->preblur_pipeline))
    {
      logError(LOG_TAG, "Failed to create PreBlur1D pipeline");
      vkDestroyShaderModule(detector->dev->device, pb_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, pb_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup AffineWarp pipeline (ASIFT batch path)
  //////////////////////////////////////
  {
    VkShaderModule aw_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/AffineWarp.comp.spv", &aw_shader_module))
    {
      logError(LOG_TAG, "Failed to create AffineWarp shader module");
      return false;
    }
    if (!vkenv_createComputePipeline2(detector->dev->device, aw_shader_module,
                                      detector->affinewarp_desc_set_layout,
                                      detector->warp_ubo_desc_set_layout,
                                      0u,
                                      &detector->affinewarp_pipeline_layout, &detector->affinewarp_pipeline))
    {
      logError(LOG_TAG, "Failed to create AffineWarp pipeline");
      vkDestroyShaderModule(detector->dev->device, aw_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, aw_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup GaussianBlur pipeline
  //////////////////////////////////////
  VkShaderModule blur_shader_module;
  if (detector->use_hardware_interp_kernel)
  {
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/GaussianBlurInterpolated.comp.spv", &blur_shader_module))
    {
      logError(LOG_TAG, "Failed to create GaussianBlurInterpolated shader module");
      return false;
    }
  }
  else if (!vkenv_createShaderModule(detector->dev->device, "shaders/GaussianBlur.comp.spv", &blur_shader_module))
  {
    logError(LOG_TAG, "Failed to create GaussianBlur shader module");
    return false;
  }
  if (!vkenv_createComputePipeline(detector->dev->device, blur_shader_module, detector->blur_desc_set_layout, sizeof(GaussianBlurPushConsts),
                                   &detector->blur_pipeline_layout, &detector->blur_pipeline))
  {
    logError(LOG_TAG, "Failed to create GaussianBlur pipeline");
    vkDestroyShaderModule(detector->dev->device, blur_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, blur_shader_module, NULL);

  //////////////////////////////////////
  // Setup DifferenceOfGaussian pipeline
  //////////////////////////////////////
  VkShaderModule dog_shader_module;
  if (!vkenv_createShaderModule(detector->dev->device, "shaders/DifferenceOfGaussian.comp.spv", &dog_shader_module))
  {
    logError(LOG_TAG, "Failed to create DifferenceOfGaussian shader module");
    return false;
  }
  if (!vkenv_createComputePipeline(detector->dev->device, dog_shader_module, detector->dog_desc_set_layout, 0, &detector->dog_pipeline_layout,
                                   &detector->dog_pipeline))
  {
    logError(LOG_TAG, "Failed to create DifferenceOfGaussian pipeline");
    vkDestroyShaderModule(detector->dev->device, dog_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, dog_shader_module, NULL);

  //////////////////////////////////////
  // Setup Downsample2x pipeline (Lowe pixel-aligned octave downsample)
  //////////////////////////////////////
  {
    VkShaderModule ds_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/Downsample2x.comp.spv", &ds_shader_module))
    {
      logError(LOG_TAG, "Failed to create Downsample2x shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, ds_shader_module, detector->downsample_desc_set_layout, sizeof(Downsample2xPushConsts),
                                     &detector->downsample_pipeline_layout, &detector->downsample_pipeline))
    {
      logError(LOG_TAG, "Failed to create Downsample2x pipeline");
      vkDestroyShaderModule(detector->dev->device, ds_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, ds_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup Upsample2xLinear pipeline (octave-0 2× linear upsample, replaces
  // vkCmdBlitImage so dispatchParallelIMAS can run on the async-compute pool).
  //////////////////////////////////////
  {
    VkShaderModule us_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/Upsample2xLinear.comp.spv", &us_shader_module))
    {
      logError(LOG_TAG, "Failed to create Upsample2xLinear shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, us_shader_module, detector->upsample_desc_set_layout, sizeof(Upsample2xLinearPushConsts),
                                     &detector->upsample_pipeline_layout, &detector->upsample_pipeline))
    {
      logError(LOG_TAG, "Failed to create Upsample2xLinear pipeline");
      vkDestroyShaderModule(detector->dev->device, us_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, us_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup SiftSeedFromInput pipeline (IMAS-fused octave-0 fast path; one
  // dispatch in place of PreBlur1D + AffineWarp + CopyImage|Upsample2xLinear).
  //////////////////////////////////////
  {
    VkShaderModule sf_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/SiftSeedFromInput.comp.spv", &sf_shader_module))
    {
      logError(LOG_TAG, "Failed to create SiftSeedFromInput shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, sf_shader_module, detector->seed_from_input_desc_set_layout, sizeof(Upsample2xLinearPushConsts),
                                     &detector->seed_from_input_pipeline_layout, &detector->seed_from_input_pipeline))
    {
      logError(LOG_TAG, "Failed to create SiftSeedFromInput pipeline");
      vkDestroyShaderModule(detector->dev->device, sf_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, sf_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup QuantizeF32ToInput pipeline (device-side R32F → R8 copy, replaces
  // host roundtrip on the on-IMAS detect path).
  //////////////////////////////////////
  {
    VkShaderModule q_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/QuantizeF32ToInput.comp.spv", &q_shader_module))
    {
      logError(LOG_TAG, "Failed to create QuantizeF32ToInput shader module");
      return false;
    }
    if (!vkenv_createComputePipeline2(detector->dev->device, q_shader_module,
                                      detector->quantize_desc_set_layout,
                                      detector->warp_ubo_desc_set_layout,
                                      0u,
                                      &detector->quantize_pipeline_layout, &detector->quantize_pipeline))
    {
      logError(LOG_TAG, "Failed to create QuantizeF32ToInput pipeline");
      vkDestroyShaderModule(detector->dev->device, q_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, q_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup ExtractKeypoints pipeline
  //////////////////////////////////////
  VkShaderModule extractkpts_shader_module;
  if (!vkenv_createShaderModule(detector->dev->device, "shaders/ExtractKeypoints.comp.spv", &extractkpts_shader_module))
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints shader module");
    return false;
  }
  if (!vkenv_createComputePipeline(detector->dev->device, extractkpts_shader_module, detector->extractkpts_desc_set_layout,
                                   sizeof(ExtractKeypointsPushConsts), &detector->extractkpts_pipeline_layout, &detector->extractkpts_pipeline))
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints pipeline");
    vkDestroyShaderModule(detector->dev->device, extractkpts_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, extractkpts_shader_module, NULL);

  //////////////////////////////////////
  // Setup ExtractKeypoints2D pipeline (2D spatial NMS, no refinement)
  //////////////////////////////////////
  VkShaderModule extractkpts_2d_shader_module;
  if (!vkenv_createShaderModule(detector->dev->device, "shaders/ExtractKeypoints2D.comp.spv", &extractkpts_2d_shader_module))
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints2D shader module");
    return false;
  }
  // Reuse the same pipeline layout as 3D (same push constants and descriptor sets)
  VkPipelineLayout extractkpts_2d_layout_unused;
  if (!vkenv_createComputePipeline(detector->dev->device, extractkpts_2d_shader_module, detector->extractkpts_desc_set_layout,
                                   sizeof(ExtractKeypointsPushConsts), &extractkpts_2d_layout_unused, &detector->extractkpts_2d_pipeline))
  {
    logError(LOG_TAG, "Failed to create ExtractKeypoints2D pipeline");
    vkDestroyShaderModule(detector->dev->device, extractkpts_2d_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, extractkpts_2d_shader_module, NULL);

  //////////////////////////////////////
  // Setup BackProjectFeatures pipeline (Phase D — GPU back-projection +
  // K·σ·σ_max parallelogram boundary filter, appended to fused IMAS+detect
  // cmd buffer). set = 0 binds the per-(slot, octave) SIFT_buffer section;
  // set = 1 binds the slot's WarpParamsUBO. No push constants.
  //////////////////////////////////////
  {
    VkShaderModule bp_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/BackProjectFeatures.comp.spv", &bp_shader_module))
    {
      logError(LOG_TAG, "Failed to create BackProjectFeatures shader module");
      return false;
    }
    if (!vkenv_createComputePipeline2(detector->dev->device, bp_shader_module,
                                      detector->backproject_desc_set_layout,
                                      detector->warp_ubo_desc_set_layout,
                                      0u,
                                      &detector->backproject_pipeline_layout,
                                      &detector->backproject_pipeline))
    {
      logError(LOG_TAG, "Failed to create BackProjectFeatures pipeline");
      vkDestroyShaderModule(detector->dev->device, bp_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, bp_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup ComputeOrientation pipeline
  //////////////////////////////////////

  VkShaderModule orientation_shader_module;
  if (!vkenv_createShaderModule(detector->dev->device, "shaders/ComputeOrientation.comp.spv", &orientation_shader_module))
  {
    logError(LOG_TAG, "Failed to create ComputeOrientation shader module");
    return false;
  }
  // Reserve push const of size uint32 to store the max number of orientation
  if (!vkenv_createComputePipeline(detector->dev->device, orientation_shader_module, detector->orientation_desc_set_layout, sizeof(uint32_t),
                                   &detector->orientation_pipeline_layout, &detector->orientation_pipeline))
  {
    logError(LOG_TAG, "Failed to create ComputeOrientation pipeline");
    vkDestroyShaderModule(detector->dev->device, orientation_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, orientation_shader_module, NULL);

  //////////////////////////////////////
  // Setup ComputeDescriptors pipeline
  //////////////////////////////////////

  VkShaderModule descriptor_shader_module;
  if (!vkenv_createShaderModule(detector->dev->device, "shaders/ComputeDescriptors.comp.spv", &descriptor_shader_module))
  {
    logError(LOG_TAG, "Failed to create ComputeDescriptors shader module");
    return false;
  }
  // Reserve push const of size uint32 to store the descriptor format (for compatibility with descriptor from other detectors)
  if (!vkenv_createComputePipeline(detector->dev->device, descriptor_shader_module, detector->descriptor_desc_set_layout, sizeof(uint32_t),
                                   &detector->descriptor_pipeline_layout, &detector->descriptor_pipeline))
  {
    logError(LOG_TAG, "Failed to create ComputeDescriptors pipeline");
    vkDestroyShaderModule(detector->dev->device, descriptor_shader_module, NULL);
    return false;
  }
  vkDestroyShaderModule(detector->dev->device, descriptor_shader_module, NULL);

  //////////////////////////////////////
  // Setup RGBAtoGray pipeline (only when use_rgba_input)
  //////////////////////////////////////
  if (detector->use_rgba_input)
  {
    VkShaderModule rgba_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/RGBAtoGray.comp.spv", &rgba_shader_module))
    {
      logError(LOG_TAG, "Failed to create RGBAtoGray shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, rgba_shader_module, detector->rgba_convert_desc_set_layout, 0,
                                     &detector->rgba_convert_pipeline_layout, &detector->rgba_convert_pipeline))
    {
      logError(LOG_TAG, "Failed to create RGBAtoGray pipeline");
      vkDestroyShaderModule(detector->dev->device, rgba_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, rgba_shader_module, NULL);
  }

  //////////////////////////////////////
  // Setup RGBtoGray pipeline (only when use_rgb_input)
  // Push constant: uint32_t image_width
  //////////////////////////////////////
  if (detector->use_rgb_input)
  {
    VkShaderModule rgb_shader_module;
    if (!vkenv_createShaderModule(detector->dev->device, "shaders/RGBtoGray.comp.spv", &rgb_shader_module))
    {
      logError(LOG_TAG, "Failed to create RGBtoGray shader module");
      return false;
    }
    if (!vkenv_createComputePipeline(detector->dev->device, rgb_shader_module, detector->rgb_convert_desc_set_layout, sizeof(uint32_t),
                                     &detector->rgb_convert_pipeline_layout, &detector->rgb_convert_pipeline))
    {
      logError(LOG_TAG, "Failed to create RGBtoGray pipeline");
      vkDestroyShaderModule(detector->dev->device, rgb_shader_module, NULL);
      return false;
    }
    vkDestroyShaderModule(detector->dev->device, rgb_shader_module, NULL);
  }

  return true;
}

static bool setupSyncObjects(vksift_SiftDetector detector)
{
  VkSemaphoreCreateInfo semaphore_create_info = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = NULL, .flags = 0};
  if (vkCreateSemaphore(detector->dev->device, &semaphore_create_info, NULL, &detector->end_of_detection_semaphore) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create end-of-detection Vulkan semaphore");
    return false;
  }

  if (detector->dev->async_transfer_available)
  {
    if (vkCreateSemaphore(detector->dev->device, &semaphore_create_info, NULL, &detector->buffer_ownership_released_by_transfer_semaphore) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create Vulkan semaphore for async transfer queue buffer ownership transfer");
      return false;
    }
  }

  // Phase E: graphics→compute sync semaphore. Created unsignaled; the
  // parallel-IMAS dispatcher signals it once per call via an empty graphics
  // submit, and the first compute-queue wave consumes it. Only meaningful on
  // devices with a dedicated async-compute queue family.
  if (detector->dev->async_compute_available)
  {
    if (vkCreateSemaphore(detector->dev->device, &semaphore_create_info, NULL, &detector->parallel_compute_start_semaphore) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create parallel-compute-start Vulkan semaphore");
      return false;
    }
  }
  else
  {
    detector->parallel_compute_start_semaphore = VK_NULL_HANDLE;
  }

  VkFenceCreateInfo fence_create_info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .pNext = NULL, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
  if (vkCreateFence(detector->dev->device, &fence_create_info, NULL, &detector->end_of_detection_fence) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to create a Vulkan fence");
    return false;
  }
  // Phase C-async: second fence dedicated to the async-compute-queue half of
  // each parallel-IMAS wave. Created signaled so the very first wave-reset
  // works without a wait. NULL when the device has no dedicated compute queue.
  if (detector->dev->async_compute_available)
  {
    if (vkCreateFence(detector->dev->device, &fence_create_info, NULL, &detector->end_of_detection_fence_compute) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to create a Vulkan fence for async-compute parallel-IMAS dispatch");
      return false;
    }
  }
  else
  {
    detector->end_of_detection_fence_compute = VK_NULL_HANDLE;
  }

  // Per-slot shader-region timestamp query pool. Created unconditionally — the
  // cmd buffer always emits vkCmdWriteTimestamp at region boundaries; we only
  // pay the host-side read+log cost when VKSIFT_PROFILE_SHADERS=1 is set.
  detector->profile_shaders = (getenv("VKSIFT_PROFILE_SHADERS") != NULL);
  memset(detector->profile_shader_acc_ns, 0, sizeof(detector->profile_shader_acc_ns));
  detector->profile_shader_n_warps = 0u;
  detector->shader_timestamp_pools = (VkQueryPool *)calloc(detector->mem->nb_pyramid_slots, sizeof(VkQueryPool));
  if (detector->shader_timestamp_pools == NULL)
  {
    logError(LOG_TAG, "Failed to allocate shader_timestamp_pools array");
    return false;
  }
  {
    VkQueryPoolCreateInfo qp_info = {
        .sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
        .queryType  = VK_QUERY_TYPE_TIMESTAMP,
        .queryCount = VKSIFT_NUM_TS};
    for (uint32_t s = 0u; s < detector->mem->nb_pyramid_slots; ++s)
    {
      if (vkCreateQueryPool(detector->dev->device, &qp_info, NULL, &detector->shader_timestamp_pools[s]) != VK_SUCCESS)
      {
        logError(LOG_TAG, "Failed to create shader timestamp query pool for slot %u", s);
        return false;
      }
    }
  }
  return true;
}

// Phase C-1: index helper for the flat per-slot per-octave arrays.
// Layout: [slot * max_nb_octaves + oct]. max_nb_octaves is the static cap
// (vksift_SiftMemory::max_nb_octaves); curr_nb_octaves <= max_nb_octaves.
static inline uint32_t slot_oct_idx(const vksift_SiftDetector det, uint32_t slot, uint32_t oct)
{
  return slot * det->mem->max_nb_octaves + oct;
}

// Index helper for the slot s in [0..N) sift buffer. Slot s writes its SIFT
// features into sift_buffer_arr[s] so concurrent waves are independent. The
// caller-supplied target_buffer_idx (vksift_dispatchSiftDetection's parameter)
// is honored for slot 0 (legacy single-slot path); slots [1..N) hard-bind to
// their own index. Phase C-2 guarantees nb_sift_buffer >= nb_pyramid_slots, so
// no clamp is needed.
static inline uint32_t slot_sift_buffer_idx(const vksift_SiftDetector det, uint32_t slot)
{
  if (slot == 0u) return det->curr_buffer_idx;
  return slot;
}

static bool writeDescriptorSets(vksift_SiftDetector detector)
{
  uint32_t N = detector->mem->nb_pyramid_slots;
  if (N == 0u) N = 1u;
  /////////////////////////////////////////////////////
  // Write bindings for the per-slot WarpParamsUBO descriptor sets
  // (set = 1 in AffineWarp.comp + QuantizeF32ToInput.comp). Each set's
  // binding 0 points at mem->slots[s].warp_params_ubo.
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorBufferInfo bi = {.buffer = detector->mem->slots[s].warp_params_ubo,
                                 .offset = 0, .range = VK_WHOLE_SIZE};
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                              .dstSet = detector->warp_ubo_desc_sets[s],
                              .dstBinding = 0, .dstArrayElement = 0,
                              .descriptorCount = 1,
                              .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                              .pBufferInfo = &bi};
    vkUpdateDescriptorSets(detector->dev->device, 1, &w, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write bindings for PreBlur1D pipeline (ASIFT σ_aa pre-blur).
  // Per-slot: slot s binds slots[s].input_image_view (sampler) +
  // slots[s].blurred_input_image_view (storage).
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorImageInfo pb_in_info = {
        .sampler = detector->image_sampler, .imageView = detector->mem->slots[s].input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo pb_out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].blurred_input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet pb_writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = detector->preblur_desc_set[s], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &pb_in_info},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = detector->preblur_desc_set[s], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &pb_out_info}};
    vkUpdateDescriptorSets(detector->dev->device, 2, pb_writes, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write bindings for AffineWarp pipeline (ASIFT batch path) — per-slot.
  // AffineWarp samples FROM the PreBlur1D output (blurred_input_image), not
  // the raw input_image, so the σ_aa pre-blur is included in the warp.
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorImageInfo aw_in_info = {
        .sampler = detector->image_sampler, .imageView = detector->mem->slots[s].blurred_input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo aw_out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].warped_input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet aw_writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = detector->affinewarp_desc_set[s], .dstBinding = 0, .dstArrayElement = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &aw_in_info},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
         .dstSet = detector->affinewarp_desc_set[s], .dstBinding = 1, .dstArrayElement = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &aw_out_info}};
    vkUpdateDescriptorSets(detector->dev->device, 2, aw_writes, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write sets for gaussian blur pipeline (per-slot per-octave).
  for (uint32_t s = 0u; s < N; ++s)
  {
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo blur_input_image_info = {
          .sampler = detector->image_sampler, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo blur_work_image_info = {
          .sampler = detector->image_sampler, .imageView = detector->mem->slots[s].blur_tmp_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo blur_output_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet blur_descriptor_writes[2];
      // First write for horizontal pass
      blur_descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         .dstSet = detector->blur_h_desc_sets[idx],
                                                         .dstBinding = 0,
                                                         .dstArrayElement = 0,
                                                         .descriptorCount = 1,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         .pImageInfo = &blur_input_image_info,
                                                         .pBufferInfo = NULL,
                                                         .pTexelBufferView = NULL};
      blur_descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         .dstSet = detector->blur_h_desc_sets[idx],
                                                         .dstBinding = 1,
                                                         .dstArrayElement = 0,
                                                         .descriptorCount = 1,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                         .pImageInfo = &blur_work_image_info,
                                                         .pBufferInfo = NULL,
                                                         .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, blur_descriptor_writes, 0, NULL);
      // Then write for vertical pass
      blur_descriptor_writes[0].dstSet = detector->blur_v_desc_sets[idx];
      blur_descriptor_writes[0].pImageInfo = &blur_work_image_info;
      blur_descriptor_writes[1].dstSet = detector->blur_v_desc_sets[idx];
      blur_descriptor_writes[1].pImageInfo = &blur_output_image_info;
      vkUpdateDescriptorSets(detector->dev->device, 2, blur_descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for Difference of Gaussian pipeline (per-slot per-octave).
  for (uint32_t s = 0u; s < N; ++s)
  {
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo dog_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo dog_output_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_DoG_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet dog_descriptor_writes[2];
      dog_descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        .dstSet = detector->dog_desc_sets[idx],
                                                        .dstBinding = 0,
                                                        .dstArrayElement = 0,
                                                        .descriptorCount = 1,
                                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                        .pImageInfo = &dog_input_image_info,
                                                        .pBufferInfo = NULL,
                                                        .pTexelBufferView = NULL};
      dog_descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                        .dstSet = detector->dog_desc_sets[idx],
                                                        .dstBinding = 1,
                                                        .dstArrayElement = 0,
                                                        .descriptorCount = 1,
                                                        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                        .pImageInfo = &dog_output_image_info,
                                                        .pBufferInfo = NULL,
                                                        .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, dog_descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for Downsample2x pipeline (per-slot per-octave). Transition i
  // reads octave_image_view_arr[i] and writes octave_image_view_arr[i+1].
  for (uint32_t s = 0u; s < N; ++s)
  {
    for (uint32_t i = 0; i + 1 < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo ds_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo ds_output_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i + 1], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet ds_writes[2];
      ds_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                            .dstSet = detector->downsample_desc_sets[idx],
                                            .dstBinding = 0,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .pImageInfo = &ds_input_image_info,
                                            .pBufferInfo = NULL,
                                            .pTexelBufferView = NULL};
      ds_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                            .dstSet = detector->downsample_desc_sets[idx],
                                            .dstBinding = 1,
                                            .dstArrayElement = 0,
                                            .descriptorCount = 1,
                                            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                            .pImageInfo = &ds_output_image_info,
                                            .pBufferInfo = NULL,
                                            .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, ds_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write set for Upsample2xLinear pipeline (per-slot). Binds:
  //   0: slots[s].warped_input_image_view (R32F image2D, AffineWarp output)
  //   1: slots[s].octave_image_view_arr[0] (R32F image2DArray, octave 0)
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorImageInfo us_in_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].warped_input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo us_out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[0], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet us_writes[2];
    us_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = detector->upsample_desc_set[s],
                                          .dstBinding = 0,
                                          .dstArrayElement = 0,
                                          .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .pImageInfo = &us_in_info,
                                          .pBufferInfo = NULL,
                                          .pTexelBufferView = NULL};
    us_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = detector->upsample_desc_set[s],
                                          .dstBinding = 1,
                                          .dstArrayElement = 0,
                                          .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .pImageInfo = &us_out_info,
                                          .pBufferInfo = NULL,
                                          .pTexelBufferView = NULL};
    vkUpdateDescriptorSets(detector->dev->device, 2, us_writes, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write set for SiftSeedFromInput pipeline (per-slot). Binds:
  //   0: slots[s].input_image_view (R8_UNORM image2D, post-Quantize)
  //   1: slots[s].octave_image_view_arr[0] (R32F image2DArray, octave 0)
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorImageInfo sf_in_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo sf_out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[0], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet sf_writes[2];
    sf_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = detector->seed_from_input_desc_set[s],
                                          .dstBinding = 0,
                                          .dstArrayElement = 0,
                                          .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .pImageInfo = &sf_in_info,
                                          .pBufferInfo = NULL,
                                          .pTexelBufferView = NULL};
    sf_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = detector->seed_from_input_desc_set[s],
                                          .dstBinding = 1,
                                          .dstArrayElement = 0,
                                          .descriptorCount = 1,
                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                          .pImageInfo = &sf_out_info,
                                          .pBufferInfo = NULL,
                                          .pTexelBufferView = NULL};
    vkUpdateDescriptorSets(detector->dev->device, 2, sf_writes, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write set for QuantizeF32ToInput pipeline (per-slot). Binds:
  //   0: slots[s].rotated_image_view (R32F, IMAS output)
  //   1: slots[s].input_image_view   (R8_UNORM, SIFT input)
  for (uint32_t s = 0u; s < N; ++s)
  {
    VkDescriptorImageInfo q_in_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].rotated_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo q_out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet q_writes[2];
    q_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                         .dstSet = detector->quantize_desc_set[s],
                                         .dstBinding = 0,
                                         .dstArrayElement = 0,
                                         .descriptorCount = 1,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                         .pImageInfo = &q_in_info,
                                         .pBufferInfo = NULL,
                                         .pTexelBufferView = NULL};
    q_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                         .dstSet = detector->quantize_desc_set[s],
                                         .dstBinding = 1,
                                         .dstArrayElement = 0,
                                         .descriptorCount = 1,
                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                         .pImageInfo = &q_out_info,
                                         .pBufferInfo = NULL,
                                         .pTexelBufferView = NULL};
    vkUpdateDescriptorSets(detector->dev->device, 2, q_writes, 0, NULL);
  }

  /////////////////////////////////////////////////////
  // Write sets for ExtractKeypoints pipeline (per-slot per-octave). The
  // descriptor binds sift_buffer_arr[slot_sift_buffer_idx(s)] so slot s writes
  // into an independent buffer; slot 0 honors curr_buffer_idx for legacy
  // compatibility with vksift_dispatchSiftDetection.
  for (uint32_t s = 0u; s < N; ++s)
  {
    const uint32_t buf_idx = slot_sift_buffer_idx(detector, s);
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo dog_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_DoG_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorBufferInfo sift_buffer_info = {.buffer = detector->mem->sift_buffer_arr[buf_idx],
                                                 .offset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[i],
                                                 .range = detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[i]};
      VkDescriptorBufferInfo indispatch_buffer_info = {.buffer = detector->mem->indirect_orientation_dispatch_buffer,
                                                       .offset = detector->mem->indirect_oridesc_offset_arr[i],
                                                       .range = sizeof(uint32_t) * 3};
      VkWriteDescriptorSet descriptor_writes[3];
      descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->extractkpts_desc_sets[idx],
                                                    .dstBinding = 0,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    .pImageInfo = &dog_input_image_info,
                                                    .pBufferInfo = NULL,
                                                    .pTexelBufferView = NULL};
      descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->extractkpts_desc_sets[idx],
                                                    .dstBinding = 1,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .pImageInfo = NULL,
                                                    .pBufferInfo = &sift_buffer_info,
                                                    .pTexelBufferView = NULL};
      descriptor_writes[2] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->extractkpts_desc_sets[idx],
                                                    .dstBinding = 2,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .pImageInfo = NULL,
                                                    .pBufferInfo = &indispatch_buffer_info,
                                                    .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 3, descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Phase D — Write sets for BackProjectFeatures (per-slot per-octave).
  // Each (slot, octave) set's binding 0 points at the same sift_buffer
  // section that the ExtractKeypoints set above writes into, so the
  // back-projection shader reads `nb_elem` + features that ExtractKeypoints
  // just emitted. Same offset / size as ExtractKeypoints binding 1.
  for (uint32_t s = 0u; s < N; ++s)
  {
    const uint32_t buf_idx = slot_sift_buffer_idx(detector, s);
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorBufferInfo sift_section_info = {
          .buffer = detector->mem->sift_buffer_arr[buf_idx],
          .offset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[i],
          .range  = detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[i]};
      VkWriteDescriptorSet bp_write = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                       .dstSet = detector->backproject_desc_sets[idx],
                                       .dstBinding = 0,
                                       .dstArrayElement = 0,
                                       .descriptorCount = 1,
                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                       .pImageInfo = NULL,
                                       .pBufferInfo = &sift_section_info,
                                       .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 1, &bp_write, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for ComputeOrientation pipeline (per-slot per-octave).
  for (uint32_t s = 0u; s < N; ++s)
  {
    const uint32_t buf_idx = slot_sift_buffer_idx(detector, s);
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo octave_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorBufferInfo sift_buffer_info = {.buffer = detector->mem->sift_buffer_arr[buf_idx],
                                                 .offset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[i],
                                                 .range = detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[i]};
      VkDescriptorBufferInfo indispatch_buffer_info = {.buffer = detector->mem->indirect_descriptor_dispatch_buffer,
                                                       .offset = detector->mem->indirect_oridesc_offset_arr[i],
                                                       .range = sizeof(uint32_t) * 3};

      VkWriteDescriptorSet descriptor_writes[3];
      descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->orientation_desc_sets[idx],
                                                    .dstBinding = 0,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    .pImageInfo = &octave_input_image_info,
                                                    .pBufferInfo = NULL,
                                                    .pTexelBufferView = NULL};
      descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->orientation_desc_sets[idx],
                                                    .dstBinding = 1,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .pImageInfo = NULL,
                                                    .pBufferInfo = &sift_buffer_info,
                                                    .pTexelBufferView = NULL};
      descriptor_writes[2] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->orientation_desc_sets[idx],
                                                    .dstBinding = 2,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .pImageInfo = NULL,
                                                    .pBufferInfo = &indispatch_buffer_info,
                                                    .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 3, descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for ComputeDescriptor pipeline (per-slot per-octave).
  for (uint32_t s = 0u; s < N; ++s)
  {
    const uint32_t buf_idx = slot_sift_buffer_idx(detector, s);
    for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
    {
      const uint32_t idx = slot_oct_idx(detector, s, i);
      VkDescriptorImageInfo octave_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].octave_image_view_arr[i], .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorBufferInfo sift_buffer_info = {.buffer = detector->mem->sift_buffer_arr[buf_idx],
                                                 .offset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[i],
                                                 .range = detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[i]};
      VkWriteDescriptorSet descriptor_writes[2];
      descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->descriptor_desc_sets[idx],
                                                    .dstBinding = 0,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                    .pImageInfo = &octave_input_image_info,
                                                    .pBufferInfo = NULL,
                                                    .pTexelBufferView = NULL};
      descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                    .dstSet = detector->descriptor_desc_sets[idx],
                                                    .dstBinding = 1,
                                                    .dstArrayElement = 0,
                                                    .descriptorCount = 1,
                                                    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                    .pImageInfo = NULL,
                                                    .pBufferInfo = &sift_buffer_info,
                                                    .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for RGBAtoGray pipeline (only when use_rgba_input). Per-slot.
  if (detector->use_rgba_input)
  {
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorImageInfo rgba_input_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].rgba_input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkDescriptorImageInfo gray_output_image_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet rgba_descriptor_writes[2];
      rgba_descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                          .dstSet = detector->rgba_convert_desc_set[s],
                                                          .dstBinding = 0,
                                                          .dstArrayElement = 0,
                                                          .descriptorCount = 1,
                                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                          .pImageInfo = &rgba_input_image_info,
                                                          .pBufferInfo = NULL,
                                                          .pTexelBufferView = NULL};
      rgba_descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                          .dstSet = detector->rgba_convert_desc_set[s],
                                                          .dstBinding = 1,
                                                          .dstArrayElement = 0,
                                                          .descriptorCount = 1,
                                                          .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                          .pImageInfo = &gray_output_image_info,
                                                          .pBufferInfo = NULL,
                                                          .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, rgba_descriptor_writes, 0, NULL);
    }
  }

  /////////////////////////////////////////////////////
  // Write sets for RGBtoGray pipeline (only when use_rgb_input). Per-slot.
  if (detector->use_rgb_input)
  {
    for (uint32_t s = 0u; s < N; ++s)
    {
      VkDescriptorBufferInfo rgb_ssbo_info = {
          .buffer = detector->mem->slots[s].rgb_input_buffer, .offset = 0, .range = VK_WHOLE_SIZE};
      VkDescriptorImageInfo rgb_gray_output_info = {
          .sampler = VK_NULL_HANDLE, .imageView = detector->mem->slots[s].input_image_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
      VkWriteDescriptorSet rgb_descriptor_writes[2];
      rgb_descriptor_writes[0] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         .dstSet = detector->rgb_convert_desc_set[s],
                                                       .dstBinding = 0,
                                                       .dstArrayElement = 0,
                                                       .descriptorCount = 1,
                                                       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                                       .pImageInfo = NULL,
                                                       .pBufferInfo = &rgb_ssbo_info,
                                                       .pTexelBufferView = NULL};
      rgb_descriptor_writes[1] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                                         .dstSet = detector->rgb_convert_desc_set[s],
                                                         .dstBinding = 1,
                                                         .dstArrayElement = 0,
                                                         .descriptorCount = 1,
                                                         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                                         .pImageInfo = &rgb_gray_output_info,
                                                         .pBufferInfo = NULL,
                                                         .pTexelBufferView = NULL};
      vkUpdateDescriptorSets(detector->dev->device, 2, rgb_descriptor_writes, 0, NULL);
    }
  }

  return true;
}

static void recCopyInputImageCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx)
{
  /////////////////////////////////////////////////
  // Copy input image — into slots[slot_idx].input_image. Also syncs
  // cached_input_image (single-instance) from the just-populated slot input.
  /////////////////////////////////////////////////
  beginMarkerRegion(detector, cmdbuf, "CopyInputImage");
  VkImageMemoryBarrier image_barrier;

  if (detector->use_rgba_input)
  {
    // RGBA path: copy staging → rgba_input_image, then compute shader → input_image
    // Transition rgba_input_image for transfer write
    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].rgba_input_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);

    // Copy staging buffer to RGBA image
    VkBufferImageCopy buffer_image_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = detector->mem->curr_input_image_width, .height = detector->mem->curr_input_image_height, .depth = 1}};
    vkCmdCopyBufferToImage(cmdbuf, detector->mem->image_staging_buffer, detector->mem->slots[slot_idx].rgba_input_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &buffer_image_region);

    // Transition rgba_input_image to GENERAL for compute read
    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].rgba_input_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);

    // Transition input_image (R8) to GENERAL for compute write
    VkImageMemoryBarrier input_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &input_barrier);

    // Dispatch RGBA→Gray compute shader
    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->rgba_convert_pipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->rgba_convert_pipeline_layout, 0, 1,
                            &detector->rgba_convert_desc_set[slot_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, (uint32_t)ceilf((float)detector->mem->curr_input_image_width / 8.f),
                  (uint32_t)ceilf((float)detector->mem->curr_input_image_height / 8.f), 1);

    // Transition input_image from compute write to shader read (for scale-space blit)
    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);
  }
  else if (detector->use_rgb_input)
  {
    // RGB path: copy staging → rgb_input_buffer (SSBO), then compute shader → input_image
    uint32_t W = detector->mem->curr_input_image_width;
    uint32_t H = detector->mem->curr_input_image_height;
    VkDeviceSize rgb_size = 3 * (VkDeviceSize)W * H;

    // Copy staging buffer → device-local RGB buffer
    VkBufferCopy buf_copy = {.srcOffset = 0, .dstOffset = 0, .size = rgb_size};
    vkCmdCopyBuffer(cmdbuf, detector->mem->image_staging_buffer, detector->mem->slots[slot_idx].rgb_input_buffer, 1, &buf_copy);

    // Barrier: transfer write to RGB buffer → shader read
    VkBufferMemoryBarrier buf_barrier = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer = detector->mem->slots[slot_idx].rgb_input_buffer,
        .offset = 0,
        .size = VK_WHOLE_SIZE};
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 1, &buf_barrier, 0, NULL);

    // Transition input_image (R8) to GENERAL for compute write
    VkImageMemoryBarrier input_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &input_barrier);

    // Dispatch RGB→Gray compute shader with push constant for image width
    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->rgb_convert_pipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->rgb_convert_pipeline_layout, 0, 1,
                            &detector->rgb_convert_desc_set[slot_idx], 0, NULL);
    vkCmdPushConstants(cmdbuf, detector->rgb_convert_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &W);
    vkCmdDispatch(cmdbuf, (uint32_t)ceilf((float)W / 8.f), (uint32_t)ceilf((float)H / 8.f), 1);

    // Transition input_image from compute write to shader read (for scale-space blit)
    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);
  }
  else
  {
    // Grayscale path: copy staging → input_image directly (original code)
    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);

    VkBufferImageCopy buffer_image_region = {
        .bufferOffset = 0,
        .bufferRowLength = 0,
        .bufferImageHeight = 0,
        .imageSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
        .imageOffset = {.x = 0, .y = 0, .z = 0},
        .imageExtent = {.width = detector->mem->curr_input_image_width, .height = detector->mem->curr_input_image_height, .depth = 1}};
    vkCmdCopyBufferToImage(cmdbuf, detector->mem->image_staging_buffer, detector->mem->slots[slot_idx].input_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                           &buffer_image_region);

    image_barrier = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &image_barrier);
  }

  // Sync cached_input_image with the just-populated input_image. The IMAS
  // pipeline samples from cached_input_image_view, so the on-IMAS detect
  // path can overwrite input_image with quantized tilted content without
  // corrupting the next warp's IMAS source.
  {
    VkImageMemoryBarrier barriers[2];
    barriers[0] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].input_image,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    barriers[1] = vkenv_genImageMemoryBarrier(
        detector->mem->cached_input_image,
        0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, NULL, 0, NULL, 2, barriers);

    VkImageCopy region = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffset = {0, 0, 0},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .dstOffset = {0, 0, 0},
        .extent = {detector->mem->curr_input_image_width, detector->mem->curr_input_image_height, 1}};
    vkCmdCopyImage(cmdbuf,
        detector->mem->slots[slot_idx].input_image, VK_IMAGE_LAYOUT_GENERAL,
        detector->mem->cached_input_image, VK_IMAGE_LAYOUT_GENERAL,
        1, &region);

    VkImageMemoryBarrier post = vkenv_genImageMemoryBarrier(
        detector->mem->cached_input_image,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &post);
  }

  endMarkerRegion(detector, cmdbuf);
}

// Records the on-IMAS variant of input-image population: instead of copying
// from the host staging buffer, runs the QuantizeF32ToInput compute shader to
// copy device-side from mem->slots[slot_idx].rotated_image (R32F, IMAS output)
// into mem->slots[slot_idx].input_image (R8_UNORM). Dispatch dims come from
// detector->quantize_width / quantize_height (legacy non-fused path) or the
// slot's indirect-dispatch buffer (fused path); both encode the same value.
static void recQuantizeImasToInputCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx)
{
  beginMarkerRegion(detector, cmdbuf, "Quantize IMAS → input");
  VkImageMemoryBarrier rotated_barrier = vkenv_genImageMemoryBarrier(
      detector->mem->slots[slot_idx].rotated_image,
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
      VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
  vkCmdPipelineBarrier(cmdbuf,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, NULL, 0, NULL, 1, &rotated_barrier);

  VkImageMemoryBarrier input_barrier = vkenv_genImageMemoryBarrier(
      detector->mem->slots[slot_idx].input_image,
      0, VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
  vkCmdPipelineBarrier(cmdbuf,
      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, NULL, 0, NULL, 1, &input_barrier);

  // Dispatch covers the FULL input_image canvas (curr_input_image_*); regions
  // outside the IMAS-written sub-rectangle (quantize_valid_*) are filled with
  // quantize_fill_value to match the host-roundtrip path's pad_tilted layout.
  // After Phase B-2, all these fields live in the WarpParamsUBO at set = 1,
  // which the host populates in dispatchDetectionCmdBuffer before submitting
  // this pre-recorded command buffer.
  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline);
  vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline_layout,
                          0, 1, &detector->quantize_desc_set[slot_idx], 0, NULL);
  vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline_layout,
                          1, 1, &detector->warp_ubo_desc_sets[slot_idx], 0, NULL);
  vkCmdDispatch(cmdbuf,
      (uint32_t)ceilf((float)detector->mem->curr_input_image_width / 8.f),
      (uint32_t)ceilf((float)detector->mem->curr_input_image_height / 8.f), 1);

  VkImageMemoryBarrier post_barrier = vkenv_genImageMemoryBarrier(
      detector->mem->slots[slot_idx].input_image,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
  vkCmdPipelineBarrier(cmdbuf,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, NULL, 0, NULL, 1, &post_barrier);
  endMarkerRegion(detector, cmdbuf);
}

// When `oct0_seed_preloaded` is true, the caller has ALREADY written into
// octave_image_arr[slot_idx][0] layer 0 (e.g. via SiftSeedFromInput in the
// IMAS-fused path). We then skip the PreBlur1D + AffineWarp + Copy/Upsample
// chain at oct_idx == 0 and jump straight to the seed-scale blur. The barrier
// going INTO the seed-scale h-blur is still emitted so the writer is visible.
static void recScaleSpaceConstructionCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_idx,
                                          bool oct0_seed_preloaded)
{
  /////////////////////////////////////////////////
  // Scale space construction (per-slot per-octave)
  /////////////////////////////////////////////////
  beginMarkerRegion(detector, cmdbuf, "Scale space construction");

  VkImageMemoryBarrier image_barriers[2];
  uint32_t nb_scales = detector->mem->nb_scales_per_octave;
  GaussianBlurPushConsts blur_push_const;
  const uint32_t oct_set_idx = slot_oct_idx(detector, slot_idx, oct_idx);

  // Octave 0: dispatch PreBlur1D (input → blurred_input) then AffineWarp
  // (blurred_input → warped_input) BEFORE the blur pipeline is bound. The
  // pre-blur applies the Morel-Yu σ_aa filter so the warp samples from an
  // anti-aliased version of the input. At σ=0 the PreBlur1D shader degenerates
  // to a pass-through copy, keeping the original identity behavior intact.
  if (oct_idx == 0 && !oct0_seed_preloaded)
  {
    beginMarkerRegion(detector, cmdbuf, "PreBlur1D + AffineWarp");

    // PreBlur1D: input_image → blurred_input_image.
    image_barriers[0] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].blurred_input_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, image_barriers);

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->preblur_pipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->preblur_pipeline_layout,
                            0, 1, &detector->preblur_desc_set[slot_idx], 0, NULL);
    PreBlur1DPushConsts pb_pc = {
        .sigma = detector->pending_blur_sigma,
        .dir_x = detector->pending_blur_dir_x,
        .dir_y = detector->pending_blur_dir_y,
    };
    vkCmdPushConstants(cmdbuf, detector->preblur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(PreBlur1DPushConsts), &pb_pc);
    vkCmdDispatch(cmdbuf,
                  (uint32_t)ceilf((float)detector->mem->curr_input_image_width  / 8.f),
                  (uint32_t)ceilf((float)detector->mem->curr_input_image_height / 8.f), 1);

    // Barrier: blurred_input_image SHADER_WRITE → SHADER_READ for AffineWarp's sampler.
    image_barriers[0] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].blurred_input_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].warped_input_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->affinewarp_pipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->affinewarp_pipeline_layout,
                            0, 1, &detector->affinewarp_desc_set[slot_idx], 0, NULL);
    // set = 1: WarpParamsUBO for slot slot_idx. Phase B-2 reads pending_warp_* +
    // curr_input_image_* from the UBO that the host populated in
    // dispatchDetectionCmdBuffer (slot 0) or vksift_dispatchFusedImasWarpForSlot
    // (slot s) before this command buffer ran.
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->affinewarp_pipeline_layout,
                            1, 1, &detector->warp_ubo_desc_sets[slot_idx], 0, NULL);
    vkCmdDispatch(cmdbuf,
                  (uint32_t)ceilf((float)detector->mem->curr_input_image_width  / 8.f),
                  (uint32_t)ceilf((float)detector->mem->curr_input_image_height / 8.f), 1);

    // Barrier: warped_input_image SHADER_WRITE → (TRANSFER_READ for the
    // vkCmdCopyImage path / SHADER_READ for the Upsample2xLinear compute path).
    // We can't tell which path runs until we compare input dims to oct-0 dims
    // below, but both paths execute the same stage transition (compute→compute
    // or compute→transfer) and both are legal on the async-compute pool. To
    // keep this simple, request the union of both access masks and target both
    // stages — the validation layer accepts this (it's a superset barrier).
    image_barriers[0] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].warped_input_image, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, image_barriers);
    endMarkerRegion(detector, cmdbuf);
  }

  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline);

  // Handle the octave first scale (copy or upsample compute dispatch from
  // warped_input_image). Two cases depending on use_upsampling:
  //   - false: octave-0 dims == input dims, so vkCmdCopyImage is 1:1
  //     bit-exact (and legal on every queue family that supports TRANSFER,
  //     including async-compute).
  //   - true:  octave-0 is 2× input — dispatch Upsample2xLinear.comp instead
  //     of vkCmdBlitImage. The blit was graphics-only; the compute shader is
  //     a bit-equivalent mirror of VK_FILTER_LINEAR + CLAMP_TO_EDGE and lets
  //     the parallel-IMAS dispatcher use the async-compute pool.
  // When `oct0_seed_preloaded`, octave_image_arr[0] layer 0 was already
  // populated by the caller (SiftSeedFromInput in the fused IMAS path); we
  // skip directly to the seed-scale blur — but still need a barrier so the
  // writer's WRITE is visible to the blur's READ.
  if (oct_idx == 0 && oct0_seed_preloaded)
  {
    image_barriers[0] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].octave_image_arr[oct_idx],
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Horizontal blur the first scale (mirrors the path below).
    blur_push_const.is_vertical = 0;
    blur_push_const.array_layer = 0;
    blur_push_const.kernel_size = detector->gaussian_kernel_sizes[0];
    memcpy(blur_push_const.kernel, detector->gaussian_kernels, sizeof(float) * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE);
    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_h_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);

    // Setup the memory access masks for vertical pass.
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Vertical blur the first scale.
    blur_push_const.is_vertical = 1;
    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_v_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);
  }
  else if (oct_idx == 0)
  {
    const uint32_t in_w  = detector->mem->curr_input_image_width;
    const uint32_t in_h  = detector->mem->curr_input_image_height;
    const uint32_t oct_w = detector->mem->octave_resolutions[oct_idx].width;
    const uint32_t oct_h = detector->mem->octave_resolutions[oct_idx].height;
    const bool same_dims = (in_w == oct_w && in_h == oct_h);

    if (same_dims)
    {
      VkImageCopy copy_region = {
          .srcSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
          .srcOffset = {0, 0, 0},
          .dstSubresource = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1},
          .dstOffset = {0, 0, 0},
          .extent = {oct_w, oct_h, 1}};
      vkCmdCopyImage(cmdbuf, detector->mem->slots[slot_idx].warped_input_image, VK_IMAGE_LAYOUT_GENERAL,
                     detector->mem->slots[slot_idx].octave_image_arr[oct_idx], VK_IMAGE_LAYOUT_GENERAL, 1, &copy_region);
    }
    else
    {
      // Upsample2xLinear: warped_input_image (image2D r32f) →
      // octave_image_arr[0] layer 0. Bit-equivalent to vkCmdBlitImage with
      // VK_FILTER_LINEAR + CLAMP_TO_EDGE.
      //
      // Need a 0 → SHADER_WRITE barrier on octave_image_arr[0] before the
      // dispatch since we are about to write to it from compute.
      image_barriers[0] = vkenv_genImageMemoryBarrier(
          detector->mem->slots[slot_idx].octave_image_arr[oct_idx], 0, VK_ACCESS_SHADER_WRITE_BIT,
          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
      vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, NULL, 0, NULL, 1, image_barriers);

      vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->upsample_pipeline);
      vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->upsample_pipeline_layout, 0, 1,
                              &detector->upsample_desc_set[slot_idx], 0, NULL);
      Upsample2xLinearPushConsts us_pc = {
          .dst_layer  = 0,
          .dst_width  = (int32_t)oct_w,
          .dst_height = (int32_t)oct_h,
          .src_width  = (int32_t)in_w,
          .src_height = (int32_t)in_h};
      vkCmdPushConstants(cmdbuf, detector->upsample_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                         sizeof(Upsample2xLinearPushConsts), &us_pc);
      vkCmdDispatch(cmdbuf, (uint32_t)ceilf((float)oct_w / 8.f), (uint32_t)ceilf((float)oct_h / 8.f), 1);
    }

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline);

    // octave_image_arr[0] is now written. For the copy path the previous write
    // was TRANSFER; for the upsample path it was COMPUTE_SHADER. The combined
    // src access mask covers both cases.
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx],
                                                    VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}); // only scale 0
    vkCmdPipelineBarrier(cmdbuf,
                         VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Horizontal blur the first scale
    blur_push_const.is_vertical = 0;
    blur_push_const.array_layer = 0;
    blur_push_const.kernel_size = detector->gaussian_kernel_sizes[0];
    memcpy(blur_push_const.kernel, detector->gaussian_kernels, sizeof(float) * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE);

    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_h_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);

    // Setup the memory access masks for vertical pass (read from temp result and write to target scale)
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}); // only scale 0
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Vertical blur the first scale
    blur_push_const.is_vertical = 1;
    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_v_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);
  }

  for (uint32_t scale_i = 1; scale_i < (nb_scales + 3); scale_i++)
  {
    // Gaussian blur from one scale to the next
    // Setup read/write access for relevant scales
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx], 0, VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, scale_i - 1, 1}); // ony prev scale
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Horizontal blur
    blur_push_const.is_vertical = 0;
    blur_push_const.array_layer = scale_i - 1;
    blur_push_const.kernel_size = detector->gaussian_kernel_sizes[scale_i];
    memcpy(blur_push_const.kernel, &detector->gaussian_kernels[scale_i * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE],
           sizeof(float) * VKSIFT_DETECTOR_MAX_GAUSSIAN_KERNEL_SIZE);

    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_h_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);
    // Change read/write acces for vertical pass
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].blur_tmp_image_arr[oct_idx], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    image_barriers[1] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, scale_i, 1}); // ony curr scale
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 2, image_barriers);

    // Vertical blur
    blur_push_const.is_vertical = 1;
    blur_push_const.array_layer = scale_i;

    vkCmdPushConstants(cmdbuf, detector->blur_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(GaussianBlurPushConsts), &blur_push_const);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->blur_pipeline_layout, 0, 1, &detector->blur_v_desc_sets[oct_set_idx], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), 1);

    // Make sure the scale image writes are available for compute
    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, scale_i, 1}); // ony curr scale
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, image_barriers);
  }

  if (oct_idx != (detector->mem->curr_nb_octaves - 1))
  {
    // Downsample seed scale of octave oct_idx → scale 0 of octave oct_idx + 1
    // via the Downsample2x compute shader: dst(i, j) = src(2*i + 1, 2*j + 1).
    // This replaces the prior vkCmdBlitImage; the blit's half-input-pixel
    // sampling offset caused Newton refinement to walk systematically +x, +y
    // at higher octaves. The compute shader is pixel-aligned with the
    // keypoint coord-conversion formula.

    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx + 1], VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, image_barriers);

    vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->downsample_pipeline);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->downsample_pipeline_layout, 0, 1,
                            &detector->downsample_desc_sets[oct_set_idx], 0, NULL);
    Downsample2xPushConsts ds_pc = {.src_layer = (int32_t)nb_scales,
                                    .dst_layer = 0,
                                    .dst_width = (int32_t)detector->mem->octave_resolutions[oct_idx + 1].width,
                                    .dst_height = (int32_t)detector->mem->octave_resolutions[oct_idx + 1].height};
    vkCmdPushConstants(cmdbuf, detector->downsample_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Downsample2xPushConsts), &ds_pc);
    vkCmdDispatch(cmdbuf, ceilf((float)ds_pc.dst_width / 8.f), ceilf((float)ds_pc.dst_height / 8.f), 1);

    image_barriers[0] = vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_image_arr[oct_idx + 1], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, image_barriers);
  }

  endMarkerRegion(detector, cmdbuf);
}

static void recDifferenceOfGaussianCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count)
{
  VkImageMemoryBarrier *image_barriers = (VkImageMemoryBarrier *)malloc(sizeof(VkImageMemoryBarrier) * oct_count);
  uint32_t nb_scales = detector->mem->nb_scales_per_octave;

  /////////////////////////////////////////////////
  // DifferenceOfGaussian (per-slot per-octave)
  /////////////////////////////////////////////////
  beginMarkerRegion(detector, cmdbuf, "DoG computation");

  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->dog_pipeline);
  // Make sure the DoG images can be written into
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    image_barriers[oct_idx - oct_begin] = vkenv_genImageMemoryBarrier(
        detector->mem->slots[slot_idx].octave_DoG_image_arr[oct_idx], 0, VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nb_scales + 2});
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, oct_count, image_barriers);

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->dog_pipeline_layout, 0, 1,
                            &detector->dog_desc_sets[slot_oct_idx(detector, slot_idx, oct_idx)], 0, NULL);
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), nb_scales + 2);
  }

  // Make the DoG images data readable for compute ops after this function
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    image_barriers[oct_idx - oct_begin] =
        vkenv_genImageMemoryBarrier(detector->mem->slots[slot_idx].octave_DoG_image_arr[oct_idx], VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                    VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                    (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, nb_scales + 2});
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, oct_count, image_barriers);

  endMarkerRegion(detector, cmdbuf);

  free(image_barriers);
}

static void recClearBufferDataCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count)
{
  beginMarkerRegion(detector, cmdbuf, "Clear buffer data");
  const uint32_t buf_idx = slot_sift_buffer_idx(detector, slot_idx);

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    vkCmdFillBuffer(cmdbuf, detector->mem->indirect_orientation_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3,
                    1);
    vkCmdFillBuffer(cmdbuf, detector->mem->indirect_descriptor_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3,
                    1);
    // Only reset the indirect dispatch buffers and the SIFT buffer section headers (sift counter and max nb sift)
    uint32_t sift_section_offset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx];
    uint32_t section_max_nb_feat = detector->mem->sift_buffers_info[buf_idx].octave_section_max_nb_feat_arr[oct_idx];
    vkCmdFillBuffer(cmdbuf, detector->mem->sift_buffer_arr[buf_idx], sift_section_offset, sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmdbuf, detector->mem->sift_buffer_arr[buf_idx], sift_section_offset + sizeof(uint32_t), sizeof(uint32_t),
                    section_max_nb_feat);

    // Set the group size x to 0 for the orientation and descriptor
    vkCmdFillBuffer(cmdbuf, detector->mem->indirect_orientation_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t), 0);
    vkCmdFillBuffer(cmdbuf, detector->mem->indirect_descriptor_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t), 0);
  }

  endMarkerRegion(detector, cmdbuf);
}

static void recExtractKeypointsCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, uint32_t oct_begin, uint32_t oct_count)
{
  /////////////////////////////////////////////////
  // Extract keypoints (per-slot per-octave)
  /////////////////////////////////////////////////
  VkBufferMemoryBarrier *buffer_barriers = (VkBufferMemoryBarrier *)malloc(sizeof(VkBufferMemoryBarrier) * oct_count * 2);
  const uint32_t buf_idx = slot_sift_buffer_idx(detector, slot_idx);
  VkBuffer sift_buffer = detector->mem->sift_buffer_arr[buf_idx];

  beginMarkerRegion(detector, cmdbuf, "ExtractKeypoints");

  // Make sure previous writes are visible
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[(oct_idx - oct_begin) * 2 + 0] =
        vkenv_genBufferMemoryBarrier(sift_buffer, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
    buffer_barriers[(oct_idx - oct_begin) * 2 + 1] =
        vkenv_genBufferMemoryBarrier(detector->mem->indirect_orientation_dispatch_buffer, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, oct_count * 2, buffer_barriers, 0,
                       NULL);

  VkPipeline kpts_pipeline = detector->use_2d_nms ? detector->extractkpts_2d_pipeline : detector->extractkpts_pipeline;
  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, kpts_pipeline);
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    ExtractKeypointsPushConsts pushconst;
    pushconst.octave_idx = (int32_t)(oct_idx) - (detector->mem->use_upsampling ? 1 : 0);
    pushconst.seed_scale_sigma = detector->seed_scale_sigma;
    pushconst.dog_threshold = detector->intensity_threshold / detector->mem->nb_scales_per_octave;
    pushconst.edge_threshold = detector->edge_threshold;
    pushconst.nb_scales = (int32_t)detector->mem->nb_scales_per_octave;
    pushconst.use_upsampling = detector->mem->use_upsampling ? 1 : 0;
    vkCmdPushConstants(cmdbuf, detector->extractkpts_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ExtractKeypointsPushConsts), &pushconst);
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->extractkpts_pipeline_layout, 0, 1,
                            &detector->extractkpts_desc_sets[slot_oct_idx(detector, slot_idx, oct_idx)],
                            0, NULL);
    // Both 2D and boundary-aware 3D NMS now dispatch ALL DoG slices
    // (nb_scales + 2). The 3D shader handles boundary slices (s == 0,
    // s == num_slices - 1) by skipping the missing-side scale comparison.
    uint32_t z_dispatch = detector->mem->nb_scales_per_octave + 2;
    vkCmdDispatch(cmdbuf, ceilf((float)(detector->mem->octave_resolutions[oct_idx].width) / 8.f),
                  ceilf((float)(detector->mem->octave_resolutions[oct_idx].height) / 8.f), z_dispatch);
  }

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] =
        vkenv_genBufferMemoryBarrier(sift_buffer, 0, VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0,
                       NULL);

  // Copy one indispatch buffer to the other
  // Prepare the access masks for the transfer
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[(oct_idx - oct_begin) * 2 + 0] = vkenv_genBufferMemoryBarrier(
        detector->mem->indirect_orientation_dispatch_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
    buffer_barriers[(oct_idx - oct_begin) * 2 + 1] =
        vkenv_genBufferMemoryBarrier(detector->mem->indirect_descriptor_dispatch_buffer, 0, VK_ACCESS_TRANSFER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
                                     VK_QUEUE_FAMILY_IGNORED, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, oct_count * 2, buffer_barriers, 0, NULL);

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    VkBufferCopy region = {.srcOffset = detector->mem->indirect_oridesc_offset_arr[oct_idx],
                           .dstOffset = detector->mem->indirect_oridesc_offset_arr[oct_idx],
                           .size = sizeof(uint32_t) * 3};
    vkCmdCopyBuffer(cmdbuf, detector->mem->indirect_orientation_dispatch_buffer, detector->mem->indirect_descriptor_dispatch_buffer, 1, &region);
  }

  // Prepare for orientation pipeline dispatch buffer for the indirect dispatch call (require specific mask)
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(
        detector->mem->indirect_orientation_dispatch_buffer, VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0, NULL);

  endMarkerRegion(detector, cmdbuf);

  free(buffer_barriers);
}

static void recComputeOrientationsCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count)
{
  /////////////////////////////////////////////////
  // Compute orientation (per-slot per-octave)
  /////////////////////////////////////////////////
  VkBufferMemoryBarrier *buffer_barriers = (VkBufferMemoryBarrier *)malloc(sizeof(VkBufferMemoryBarrier) * oct_count);
  const uint32_t buf_idx = slot_sift_buffer_idx(detector, slot_idx);
  VkBuffer sift_buffer = detector->mem->sift_buffer_arr[buf_idx];

  beginMarkerRegion(detector, cmdbuf, "ComputeOrientation");
  // Prepare the descriptor pipeline indirect dispatch buffer for writes access and make sure previous writes are visible
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(detector->mem->indirect_descriptor_dispatch_buffer, VK_ACCESS_TRANSFER_WRITE_BIT,
                                                                        VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                                                        detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0, NULL);

  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->orientation_pipeline);
  vkCmdPushConstants(cmdbuf, detector->orientation_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &detector->max_nb_orientations);
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->orientation_pipeline_layout, 0, 1,
                            &detector->orientation_desc_sets[slot_oct_idx(detector, slot_idx, oct_idx)],
                            0, NULL);
    vkCmdDispatchIndirect(cmdbuf, detector->mem->indirect_orientation_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx]);
  }

  // Make sure writes are visible for future compute shaders
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] =
        vkenv_genBufferMemoryBarrier(sift_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
                                     detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0,
                       NULL);

  // Prepare for descriptor pipeline dispatch buffer for the indirect dispatch call (require specific mask)
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(
        detector->mem->indirect_descriptor_dispatch_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_INDIRECT_COMMAND_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
        VK_QUEUE_FAMILY_IGNORED, detector->mem->indirect_oridesc_offset_arr[oct_idx], sizeof(uint32_t) * 3);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0, NULL);

  endMarkerRegion(detector, cmdbuf);

  free(buffer_barriers);
}

static void recComputeDestriptorsCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count)
{
  /////////////////////////////////////////////////
  // Compute descriptor (per-slot per-octave)
  /////////////////////////////////////////////////
  beginMarkerRegion(detector, cmdbuf, "ComputeDescriptors");
  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->descriptor_pipeline);

  vkCmdPushConstants(cmdbuf, detector->descriptor_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &detector->use_vlfeat_format);
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->descriptor_pipeline_layout, 0, 1,
                            &detector->descriptor_desc_sets[slot_oct_idx(detector, slot_idx, oct_idx)],
                            0, NULL);
    vkCmdDispatchIndirect(cmdbuf, detector->mem->indirect_descriptor_dispatch_buffer, detector->mem->indirect_oridesc_offset_arr[oct_idx]);
  }
  endMarkerRegion(detector, cmdbuf);
}

static void recCopySIFTCountCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count)
{
  /////////////////////////////////////////////////
  // Copy SIFT count to staging (per-slot sift buffer)
  /////////////////////////////////////////////////
  VkBufferMemoryBarrier *buffer_barriers = (VkBufferMemoryBarrier *)malloc(sizeof(VkBufferMemoryBarrier) * oct_count);
  const uint32_t buf_idx = slot_sift_buffer_idx(detector, slot_idx);
  VkBuffer sift_buffer = detector->mem->sift_buffer_arr[buf_idx];

  beginMarkerRegion(detector, cmdbuf, "CopySiftCount");

  // Only copy the number of detected SIFT features to the staging buffer (accessible by host)
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    buffer_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(
        sift_buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
        detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
  }
  vkCmdPipelineBarrier(cmdbuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, oct_count, buffer_barriers, 0, NULL);

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    VkBufferCopy sift_copy_region = {.srcOffset = detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
                                     .dstOffset = sizeof(uint32_t) * oct_idx,
                                     .size = sizeof(uint32_t)};
    vkCmdCopyBuffer(cmdbuf, sift_buffer, detector->mem->sift_count_staging_buffer_arr[buf_idx], 1, &sift_copy_region);
  }
  endMarkerRegion(detector, cmdbuf);

  free(buffer_barriers);
}

// Phase D — record GPU back-projection + boundary filter dispatches into the
// fused command buffer. One dispatch per octave: each section has its own
// per-(slot, octave) descriptor set (`backproject_desc_sets[slot_oct_idx]`)
// that bakes the section offset/range. Workgroup count = (1, 1, 1) with
// local_size_x = 64; the shader stride-loops over feature indices so any
// reasonable section size is handled by a single workgroup.
//
// MUST run AFTER recCopySIFTCountCmds — the count copy reads the original
// nb_elem header from each section, which back-projection doesn't touch
// (only `data[i].x`, `.y`, and possibly `.octave_idx` get rewritten). Order
// is enforced by recCopySIFTCountCmds' TRANSFER_BIT → SHADER_READ_BIT
// barrier sequence plus an additional explicit barrier here.
static void recBackProjectFeaturesCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx,
                                       const uint32_t oct_begin, const uint32_t oct_count)
{
  const uint32_t buf_idx = slot_sift_buffer_idx(detector, slot_idx);
  VkBuffer sift_buffer = detector->mem->sift_buffer_arr[buf_idx];

  beginMarkerRegion(detector, cmdbuf, "BackProjectFeatures");

  // Wait for ExtractKeypoints' writes (and the subsequent count-copy reads)
  // to land before we rewrite x/y/octave_idx in the same section. The count
  // copy uses TRANSFER_READ_BIT; back-projection's shader writes need to
  // sync against both that and any previous SHADER_WRITE_BIT.
  VkBufferMemoryBarrier *pre_barriers = (VkBufferMemoryBarrier *)malloc(sizeof(VkBufferMemoryBarrier) * oct_count);
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    pre_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(
        sift_buffer,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT,
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
        detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
  }
  vkCmdPipelineBarrier(cmdbuf,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      0, 0, NULL, oct_count, pre_barriers, 0, NULL);
  free(pre_barriers);

  vkCmdBindPipeline(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE, detector->backproject_pipeline);
  // set = 1 is the shared per-slot warp_params_ubo, same descriptor set used by
  // every IMAS-chain shader. Bound once for all octaves of this slot.
  vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                          detector->backproject_pipeline_layout, 1, 1,
                          &detector->warp_ubo_desc_sets[slot_idx], 0, NULL);

  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    vkCmdBindDescriptorSets(cmdbuf, VK_PIPELINE_BIND_POINT_COMPUTE,
                            detector->backproject_pipeline_layout, 0, 1,
                            &detector->backproject_desc_sets[slot_oct_idx(detector, slot_idx, oct_idx)],
                            0, NULL);
    // local_size_x=64 + a stride-loop inside the shader → one workgroup per
    // section regardless of feature count. ~30K features/octave × 64-thread
    // workgroup = 470 loop iters, fully bandwidth-bound (no contention).
    vkCmdDispatch(cmdbuf, 1u, 1u, 1u);
  }

  // Make x/y/octave_idx writes visible to the host transfer reads that
  // come next (the JL driver downloads the buffer after the fence).
  VkBufferMemoryBarrier *post_barriers = (VkBufferMemoryBarrier *)malloc(sizeof(VkBufferMemoryBarrier) * oct_count);
  for (uint32_t oct_idx = oct_begin; oct_idx < (oct_begin + oct_count); oct_idx++)
  {
    post_barriers[oct_idx - oct_begin] = vkenv_genBufferMemoryBarrier(
        sift_buffer, VK_ACCESS_SHADER_WRITE_BIT,
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        detector->mem->sift_buffers_info[buf_idx].octave_section_offset_arr[oct_idx],
        detector->mem->sift_buffers_info[buf_idx].octave_section_size_arr[oct_idx]);
  }
  vkCmdPipelineBarrier(cmdbuf,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
      0, 0, NULL, oct_count, post_barriers, 0, NULL);
  free(post_barriers);

  endMarkerRegion(detector, cmdbuf);
}

static void recBufferOwnershipTransferCmds(vksift_SiftDetector detector, VkCommandBuffer cmdbuf, uint32_t slot_idx, const uint32_t oct_begin, const uint32_t oct_count,
                                           const uint32_t src_queue_family_idx, const uint32_t dst_queue_family_idx, VkPipelineStageFlags src_stage,
                                           VkPipelineStageFlags dst_stage)
{
  // No-op. sift_buffer_arr is now created CONCURRENT across (general,
  // async-compute, async-transfer) by multi_queue_share_info — see
  // sift_memory.c. Ownership-transfer barriers are unnecessary on CONCURRENT
  // resources and are prohibited by VUID-VkBufferMemoryBarrier-None-09050
  // (src/dst queue family indices must be VK_QUEUE_FAMILY_IGNORED).
  // Cross-queue memory + execution dependency on the legacy single-queue
  // detection path still holds because vkQueueSubmit's semaphore signal/wait
  // establishes both per the Vulkan synchronization spec.
  (void)detector; (void)cmdbuf; (void)slot_idx;
  (void)oct_begin; (void)oct_count;
  (void)src_queue_family_idx; (void)dst_queue_family_idx;
  (void)src_stage; (void)dst_stage;
}

// =============================================================================
// Phase B-3 — fused IMAS-chain + Quantize + SIFT-detect command buffer per slot
// =============================================================================
//
// Records the union of vksift_runImasWarp's chain (sift_imas.c) and
// recordCommandBuffers's detection_command_buffer_from_imas branch into a
// single pre-recorded primary command buffer for `slot_idx`. Re-recorded
// whenever the memory layout changes (input resolution → pyramid resize) or
// the IMAS pipeline first comes up.
//
// All IMAS-chain dispatches use vkCmdDispatchIndirect against the slot's
// host-mapped dispatch_buffer (SlotDispatchBuffer layout in sift_warp_ubo.h);
// host fills the group counts per warp before submitting. SIFT-detect
// dispatches stay direct — their canvas is curr_input_image_* which is stable
// across warps. The SIFT detect descriptor sets bind slots[0]'s image views
// (Phase A holdover); slot_idx > 0 is allocated but not yet exercised here.
// Phase C-async: the `cmd` parameter selects which command buffer the
// recording lands in — the general-pool buffer
// (detector->fused_imas_detect_command_buffer[slot_idx]) is used by the
// single-queue and general-queue-half of the parallel path, the async-compute
// pool mirror (fused_imas_detect_command_buffer_compute[slot_idx]) is used by
// the compute-queue-half of the parallel path. The recorded content is
// identical for the same slot; only the owning queue family differs.
static bool recFusedImasDetectCmdsForSlot(vksift_SiftDetector detector, VkCommandBuffer cmd, uint32_t slot_idx)
{
  // No-op until the IMAS pipeline has been lazily created by the first
  // fused-dispatch call. The cmd buffer stays empty (but valid — we still
  // begin/end it so it's safe to leave allocated).
  if (detector->imas_pipeline_ref == NULL)
  {
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return false;
    if (vkEndCommandBuffer(cmd) != VK_SUCCESS) return false;
    return true;
  }

  struct vksift_ImasPipeline_T *imas = detector->imas_pipeline_ref;
  vksift_SiftPyramidSlot *slot = &detector->mem->slots[slot_idx];

  VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  if (vkBeginCommandBuffer(cmd, &bi) != VK_SUCCESS) return false;

  // Shader-region timestamp pool: reset to UNDEFINED then stamp ts[0] = START.
  // Subsequent vkCmdWriteTimestamp calls at region boundaries below capture
  // GPU pipeline wallclock for {imas, quant, scales, dog, extract, oridesc,
  // bp+count}. Host reads them in dispatchParallelIMAS when profile_shaders.
  vkCmdResetQueryPool(cmd, detector->shader_timestamp_pools[slot_idx], 0, VKSIFT_NUM_TS);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_START);

  // Phase E defensive: HOST_WRITE → UNIFORM/SHADER/INDIRECT_COMMAND_READ
  // barrier at the head of every fused cmd buffer. vkQueueSubmit's implicit
  // host-write visibility should cover this, but on the async-compute queue
  // we've observed per-slot UBO + indirect dispatch_buffer reads behaving as
  // if the host's per-wave update isn't visible — adding an explicit barrier
  // is cheap belt-and-suspenders and matches the IMAS chain on the general
  // pool's contract.
  {
    VkMemoryBarrier host_to_compute = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_HOST_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT};
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
        0, 1, &host_to_compute, 0, NULL, 0, NULL);
  }

  // ---- Acquire SIFT buffer ownership if async transfer is enabled ----
  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->async_transfer_queues_family_idx,
                                   detector->dev->general_queues_family_idx,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }

  // ---- Layout transitions for IMAS scratch images + cached input ----
  // cached_input_image must be GENERAL for sampler reads (single shared
  // resource — not per-slot). rotated/tilted scratch start UNDEFINED → GENERAL
  // (contents overwritten by IMAS chain).
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(
        detector->mem->cached_input_image,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &b);
  }
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->rotated_image,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
  }
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->tilted_image,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
  }

  // The slot's WarpParamsUBO descriptor (set = 1) is the same for every IMAS
  // shader. Bind it once via each pipeline layout below — we still have to
  // re-bind through each pipeline_layout since they're distinct VkPipelineLayout
  // objects (Vulkan binds descriptor sets per pipeline layout).
  VkDescriptorSet ubo_set = imas->warp_ubo_desc_sets[slot_idx];

  beginMarkerRegion(detector, cmd, "IMAS chain (indirect)");

  // ---- 1. AffineWarp (rotate) : cached_input_image → slot.rotated_image ----
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->warp_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->warp_pipeline_layout,
                          0, 1, &imas->warp_set[slot_idx], 0, NULL);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->warp_pipeline_layout,
                          1, 1, &ubo_set, 0, NULL);
  vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                        offsetof(SlotDispatchBuffer, affinewarp));
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->rotated_image,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
  }

  // ---- 2. GaussBlur1D vertical : slot.rotated_image → slot.tilted_image ----
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->blur_pipeline);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->blur_pipeline_layout,
                          0, 1, &imas->blur_set[slot_idx], 0, NULL);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->blur_pipeline_layout,
                          1, 1, &ubo_set, 0, NULL);
  vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                        offsetof(SlotDispatchBuffer, gaussblur));
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->tilted_image,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &b);
  }

  // ---- 3-4. Finvspline{Row,Col} : in-place per-row+col IIR on tilted_image.
  // Required when Fproj uses cubic interpolation; skipped under bilinear mode
  // (VKSIFT_IMAS_BILINEAR=1) — bilinear samples directly from the blurred
  // image without the spline pre-filter. Saves 2 full-image dispatches per warp.
  if (!detector->use_bilinear_fproj)
  {
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_row_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_pipeline_layout,
                            0, 1, &imas->finvspline_row_set[slot_idx], 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_pipeline_layout,
                            1, 1, &ubo_set, 0, NULL);
    vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                          offsetof(SlotDispatchBuffer, finvspline_row));
    {
      VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->tilted_image,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, NULL, 0, NULL, 1, &b);
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_col_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_pipeline_layout,
                            0, 1, &imas->finvspline_col_set[slot_idx], 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->finvspline_pipeline_layout,
                            1, 1, &ubo_set, 0, NULL);
    vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                          offsetof(SlotDispatchBuffer, finvspline_col));
    {
      VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(slot->tilted_image,
          VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
          VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
      vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 0, NULL, 0, NULL, 1, &b);
    }
  }

  // ---- 5. Fproj{Cubic|Bilinear}Y : slot.tilted_image → slot.rotated_image ----
  VkPipeline fproj_pipe = detector->use_bilinear_fproj
                          ? imas->fproj_bilinear_pipeline
                          : imas->fproj_pipeline;
  vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, fproj_pipe);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->fproj_pipeline_layout,
                          0, 1, &imas->fproj_set[slot_idx], 0, NULL);
  vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, imas->fproj_pipeline_layout,
                          1, 1, &ubo_set, 0, NULL);
  vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                        offsetof(SlotDispatchBuffer, fproj));
  endMarkerRegion(detector, cmd);

  // Timestamp: end of IMAS chain (rotate + finvspline + fproj + warp).
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_IMAS);

  // ---- 6. Quantize : slot.rotated_image (R32F) → slot.input_image (R8) ----
  // Reuses recQuantizeImasToInputCmds's barrier sequence + indirect dispatch.
  beginMarkerRegion(detector, cmd, "Quantize IMAS → input (fused)");
  {
    VkImageMemoryBarrier rotated_barrier = vkenv_genImageMemoryBarrier(
        slot->rotated_image,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &rotated_barrier);

    VkImageMemoryBarrier input_barrier = vkenv_genImageMemoryBarrier(
        slot->input_image,
        0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &input_barrier);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline_layout,
                            0, 1, &detector->quantize_desc_set[slot_idx], 0, NULL);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, detector->quantize_pipeline_layout,
                            1, 1, &detector->warp_ubo_desc_sets[slot_idx], 0, NULL);
    vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                          offsetof(SlotDispatchBuffer, quantize));

    VkImageMemoryBarrier post_barrier = vkenv_genImageMemoryBarrier(
        slot->input_image,
        VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &post_barrier);
  }
  endMarkerRegion(detector, cmd);

  // Timestamp: end of Quantize.
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_QUANT);

  // ---- 6b. (Optional, VKSIFT_FUSED_OCT0=1) Seed-from-input fast path. ----
  // Replaces the PreBlur1D + AffineWarp + (CopyImage|Upsample2xLinear) chain
  // at oct_idx==0 with a single SiftSeedFromInput dispatch that writes
  // input_image (R8) directly into octave_image_arr[0] layer 0 (R32F),
  // with optional 2× upsample built in. Also corrects a latent bug where the
  // OLD path's AffineWarp double-applied the IMAS rotation matrix — feature
  // counts change vs. the default path, so this is opt-in pending validation.
  if (detector->fused_oct0_enabled)
  {
    const uint32_t in_w  = detector->mem->curr_input_image_width;
    const uint32_t in_h  = detector->mem->curr_input_image_height;
    const uint32_t oct_w = detector->mem->octave_resolutions[0].width;
    const uint32_t oct_h = detector->mem->octave_resolutions[0].height;

    VkImageMemoryBarrier oct0_in = vkenv_genImageMemoryBarrier(
        slot->octave_image_arr[0], 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &oct0_in);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, detector->seed_from_input_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            detector->seed_from_input_pipeline_layout, 0, 1,
                            &detector->seed_from_input_desc_set[slot_idx], 0, NULL);
    Upsample2xLinearPushConsts sf_pc = {
        .dst_layer  = 0,
        .dst_width  = (int32_t)oct_w,
        .dst_height = (int32_t)oct_h,
        .src_width  = (int32_t)in_w,
        .src_height = (int32_t)in_h};
    vkCmdPushConstants(cmd, detector->seed_from_input_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(Upsample2xLinearPushConsts), &sf_pc);
    // Indirect dispatch via the slot's SlotDispatchBuffer.seed_from_input
    // field (populated host-side per warp in vksift_fillFusedWarpState).
    // The cmd buffer recording becomes independent of oct-0 dims, matching
    // the rest of the IMAS chain's indirect pattern.
    vkCmdDispatchIndirect(cmd, slot->dispatch_buffer,
                          offsetof(SlotDispatchBuffer, seed_from_input));
  }

  // ---- 7. SIFT detect chain (per-slot) ----
  // Direct dispatches; canvas is curr_input_image_* which is stable across
  // warps. recExtractKeypointsCmds / recCopySIFTCountCmds use slot_idx to
  // pick sift_buffer_arr[slot_idx] (slot 0 honors curr_buffer_idx), so
  // concurrent waves write to independent feature buffers.
  recClearBufferDataCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
  for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
  {
    // When fused_oct0_enabled, the SiftSeedFromInput dispatch above already
    // wrote octave_image_arr[0] layer 0, so skip the PreBlur/Warp/Copy chain.
    recScaleSpaceConstructionCmds(detector, cmd, slot_idx, i,
                                  /*oct0_seed_preloaded=*/(i == 0 && detector->fused_oct0_enabled));
  }
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_SCALES);

  recDifferenceOfGaussianCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_DOG);

  recExtractKeypointsCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_EXTRACT);

  if (!detector->detection_only)
  {
    recComputeOrientationsCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
    recComputeDestriptorsCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
  }
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_AFTER_ORIDESC);

  recCopySIFTCountCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);

  // ---- Phase D — GPU back-projection + boundary filter ----
  // After ExtractKeypoints + count-copy, rewrite each emitted feature's
  // (x, y) from tilted-frame → input-frame via the WarpParamsUBO's bp_*
  // matrix, and reject features whose K·σ·σ_max neighbourhood overlaps
  // the parallelogram edge (octave_idx = -1).
  recBackProjectFeaturesCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves);
  vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                      detector->shader_timestamp_pools[slot_idx], VKSIFT_TS_END);

  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, cmd, slot_idx, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->general_queues_family_idx,
                                   detector->dev->async_transfer_queues_family_idx,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT,
                                   VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  }

  if (vkEndCommandBuffer(cmd) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to record fused IMAS+detect command buffer (slot %u)", slot_idx);
    return false;
  }
  return true;
}

static bool recordCommandBuffers(vksift_SiftDetector detector)
{
  VkCommandBufferBeginInfo begin_info = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO, .flags = 0, .pInheritanceInfo = NULL};

  /////////////////////////////////////////////////////
  // If async transfer queue is used, record ownership transfer command buffers
  /////////////////////////////////////////////////////
  if (detector->dev->async_transfer_available)
  {
    if (vkBeginCommandBuffer(detector->release_buffer_ownership_command_buffer, &begin_info) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to begin the release-buffer-ownership command buffer recording");
      return false;
    }
    // Legacy single-slot ownership transfer cmd buffers use slot 0's sift buffer
    // (= sift_buffer_arr[curr_buffer_idx]). The fused per-slot path emits its
    // own ownership barriers inline against the correct slot's buffer.
    recBufferOwnershipTransferCmds(detector, detector->release_buffer_ownership_command_buffer, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->async_transfer_queues_family_idx, detector->dev->general_queues_family_idx,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (vkEndCommandBuffer(detector->release_buffer_ownership_command_buffer) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to record release-buffer-ownership command buffer");
      return false;
    }

    if (vkBeginCommandBuffer(detector->acquire_buffer_ownership_command_buffer, &begin_info) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to begin the acquire-buffer-ownership command buffer recording");
      return false;
    }
    recBufferOwnershipTransferCmds(detector, detector->acquire_buffer_ownership_command_buffer, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->general_queues_family_idx, detector->dev->async_transfer_queues_family_idx,
                                   VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
    if (vkEndCommandBuffer(detector->acquire_buffer_ownership_command_buffer) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to record acquire-buffer-ownership command buffer");
      return false;
    }
  }

  /////////////////////////////////////////////////////
  // Write the detection command buffer (single queue version)
  /////////////////////////////////////////////////////
  if (vkBeginCommandBuffer(detector->detection_command_buffer, &begin_info) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to begin the command buffer recording");
    return false;
  }

  // We start using the SIFT buffer, is the async transfer is used we need to acquire the buffer ownership before using it.
  // Legacy non-fused single-slot path uses slot 0 (slot_idx = 0u).
  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->async_transfer_queues_family_idx, detector->dev->general_queues_family_idx,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }

  // Clear buffer data
  recClearBufferDataCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);
  // Copy input image
  recCopyInputImageCmds(detector, detector->detection_command_buffer, 0u);

  // Scale space construction
  for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
  {
    // Construct each octave
    recScaleSpaceConstructionCmds(detector, detector->detection_command_buffer, 0u, i, /*oct0_seed_preloaded=*/false);
  }

  // Compute difference of Gaussian (on full range to synchronize every octave with a single barrier)
  recDifferenceOfGaussianCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);

  // Extract extrema (keypoints) from DoG images
  recExtractKeypointsCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);

  if (!detector->detection_only)
  {
    // For the main orientations of each keypoint (this creates new keypoints if there's more than one orientation)
    recComputeOrientationsCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);

    // For each oriented keypoint compute its descriptor
    recComputeDestriptorsCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);
  }

  // Copy the number of found keypoints to the sift_count staging buffer
  // (so that when the CPU want to download the result it can download only the number of SIFT found with a custom command buffer)
  recCopySIFTCountCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves);

  // No more operation with the buffer we can release the buffer ownership if needed
  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, detector->detection_command_buffer, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->general_queues_family_idx, detector->dev->async_transfer_queues_family_idx,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  }

  if (vkEndCommandBuffer(detector->detection_command_buffer) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to record command buffer");
    return false;
  }

  /////////////////////////////////////////////////////
  // Write the on-IMAS variant — same as above, but recQuantizeImasToInputCmds
  // replaces recCopyInputImageCmds (no host roundtrip — reads rotated_image
  // device-side via the quantize shader).
  /////////////////////////////////////////////////////
  if (vkBeginCommandBuffer(detector->detection_command_buffer_from_imas, &begin_info) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to begin the on-IMAS detection command buffer recording");
    return false;
  }
  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->async_transfer_queues_family_idx, detector->dev->general_queues_family_idx,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
  }
  recClearBufferDataCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
  recQuantizeImasToInputCmds(detector, detector->detection_command_buffer_from_imas, 0u);
  for (uint32_t i = 0; i < detector->mem->curr_nb_octaves; i++)
  {
    recScaleSpaceConstructionCmds(detector, detector->detection_command_buffer_from_imas, 0u, i, /*oct0_seed_preloaded=*/false);
  }
  recDifferenceOfGaussianCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
  recExtractKeypointsCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
  if (!detector->detection_only)
  {
    recComputeOrientationsCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
    recComputeDestriptorsCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
  }
  recCopySIFTCountCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves);
  if (detector->dev->async_transfer_available)
  {
    recBufferOwnershipTransferCmds(detector, detector->detection_command_buffer_from_imas, 0u, 0, detector->mem->curr_nb_octaves,
                                   detector->dev->general_queues_family_idx, detector->dev->async_transfer_queues_family_idx,
                                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
  }
  if (vkEndCommandBuffer(detector->detection_command_buffer_from_imas) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to record on-IMAS detection command buffer");
    return false;
  }

  /////////////////////////////////////////////////////
  // Phase B-3: record fused IMAS-chain + Quantize + SIFT-detect cmd buffer
  // per slot. The recording is no-op until imas_pipeline_ref is wired up by
  // vksift_detectFeaturesFusedImas's lazy-init path (see vulkansift.c). After
  // that, every call to recordCommandBuffers re-records the fused buffers as
  // well so they pick up new pyramid resolutions / IMAS pipeline objects.
  /////////////////////////////////////////////////////
  for (uint32_t s = 0u; s < detector->mem->nb_pyramid_slots; ++s)
  {
    if (!recFusedImasDetectCmdsForSlot(detector, detector->fused_imas_detect_command_buffer[s], s)) return false;
  }

  // Phase C-async: also record the same content into the async-compute-pool
  // mirror so the parallel-IMAS dispatcher can submit half of every wave on
  // async_compute_queues[0]. The shader pipelines + descriptor bindings are
  // identical; only the command buffer's owning queue family differs.
  if (detector->dev->async_compute_available)
  {
    for (uint32_t s = 0u; s < detector->mem->nb_pyramid_slots; ++s)
    {
      if (!recFusedImasDetectCmdsForSlot(detector, detector->fused_imas_detect_command_buffer_compute[s], s)) return false;
    }
  }

  return true;
}

bool vksift_createSiftDetector(vkenv_Device device, vksift_SiftMemory memory, vksift_SiftDetector *detector_ptr, const vksift_Config *config)
{
  assert(device != NULL);
  assert(memory != NULL);
  assert(detector_ptr != NULL);
  assert(*detector_ptr == NULL);
  assert(config != NULL);
  *detector_ptr = (vksift_SiftDetector)malloc(sizeof(struct vksift_SiftDetector_T));
  vksift_SiftDetector detector = *detector_ptr;
  memset(detector, 0, sizeof(struct vksift_SiftDetector_T));

  // Store parent device and memory
  detector->dev = device;
  detector->mem = memory;

  // Assign queues
  detector->general_queue = device->general_queues[0];
  if (device->async_transfer_available)
  {
    detector->async_ownership_transfer_queue = device->async_transfer_queues[1]; // queue 0 used by SiftMemory only
  }

  // Retrieve config
  detector->use_hardware_interp_kernel = config->use_hardware_interpolated_blur;
  detector->input_blur_level = config->input_image_blur_level;
  detector->seed_scale_sigma = config->seed_scale_sigma;
  detector->intensity_threshold = config->intensity_threshold;
  detector->edge_threshold = config->edge_threshold;
  detector->max_nb_orientations = config->max_nb_orientation_per_keypoint;
  detector->use_vlfeat_format = config->descriptor_format == VKSIFT_DESCRIPTOR_FORMAT_VLFEAT ? 1u : 0u;
  detector->detection_only = config->detection_only;
  detector->use_2d_nms = config->use_2d_nms;
  detector->use_rgba_input = config->use_rgba_input;
  detector->use_rgb_input = config->use_rgb_input;

  detector->curr_buffer_idx = 0u; // Default target buffer is 0 (always available)

  // Fused IMAS chain: bilinear Fproj is the default — skips FinvsplineRow +
  // FinvsplineCol and uses FprojBilinearY in place of FprojCubicY. Measured
  // ~1.9× wallclock speedup on IMAS-25 with sum_kept change <0.3% on the
  // find_boards test image (kept-feature count is gated by NMS + boundary
  // filter which absorb the cubic-vs-bilinear extrema-population difference).
  // Set VKSIFT_IMAS_BILINEAR=0 to force cubic. Read once at init.
  {
    const char *s = getenv("VKSIFT_IMAS_BILINEAR");
    detector->use_bilinear_fproj = (s == NULL) || (atoi(s) != 0);
  }

  // VKSIFT_FUSED_OCT0 controls the octave-0 fast path. *Default ON* — verified
  // with a single-blob synthetic test that the OLD (off) path double-applies
  // the IMAS rotation matrix at oct_idx=0 of the fused IMAS chain, so detected
  // features back-project to wrong positions (~hundreds of pixels off truth)
  // for any θ≠0 warp. Symptom: spurious "board-grid" artifacts. With this on,
  // every IMAS-25 warp localises the blob within 1-2px (vs 261-1727px before).
  // Set VKSIFT_FUSED_OCT0=0 to fall back to the legacy buggy path.
  {
    const char *s = getenv("VKSIFT_FUSED_OCT0");
    detector->fused_oct0_enabled = (s == NULL) || (atoi(s) != 0);
  }

  // PreBlur1D defaults: σ=0 means the shader degenerates to a pass-through
  // copy of input_image (no anti-alias). Direction is irrelevant when σ=0.
  detector->pending_blur_sigma = 0.0f;
  detector->pending_blur_dir_x = 0.0f;
  detector->pending_blur_dir_y = 1.0f;
  detector->pending_blur_dirty = false;

  // AffineWarp pending matrix defaults to identity (no warp).
  detector->pending_warp_a11 = 1.0f; detector->pending_warp_a12 = 0.0f; detector->pending_warp_a13 = 0.0f;
  detector->pending_warp_a21 = 0.0f; detector->pending_warp_a22 = 1.0f; detector->pending_warp_a23 = 0.0f;
  detector->pending_warp_fill = 0.0f;
  detector->pending_warp_dirty = false;

  // Try to find GPU debug marker functions
  getGPUDebugMarkerFuncs(detector);
  // Compute the Gaussian kernels used to build the scalespaces
  setupGaussianKernels(detector);

  if (setupCommandPools(detector) && allocateCommandBuffers(detector) && setupImageSampler(detector) && prepareDescriptorSets(detector) &&
      setupComputePipelines(detector) && setupSyncObjects(detector) && writeDescriptorSets(detector) && recordCommandBuffers(detector))
  {
    return true;
  }
  else
  {
    logError(LOG_TAG, "Failed to setup the SiftDetector instance");
    return false;
  }
}

static bool dispatchDetectionCmdBuffer(vksift_SiftDetector detector,
                                       const uint32_t target_buffer_idx,
                                       const bool memory_layout_updated,
                                       VkCommandBuffer *cmd_buffer_to_submit)
{
  // We need to setup the descriptor sets and command buffers if the input resolution, target buffer,
  // or pending PreBlur changed. (PreBlur1D still uses push constants so the
  // cmd buffer needs re-record when the σ changes. AffineWarp + Quantize no
  // longer require re-record since their params come from the WarpParamsUBO.)
  if (memory_layout_updated || detector->curr_buffer_idx != target_buffer_idx ||
      detector->pending_blur_dirty)
  {
    detector->curr_buffer_idx = target_buffer_idx;
    writeDescriptorSets(detector);
    recordCommandBuffers(detector);
    detector->pending_blur_dirty = false;
  }
  // pending_warp_dirty is no longer a re-record trigger — the affine matrix
  // is read from the UBO every dispatch. Clear it for callers tracking it.
  detector->pending_warp_dirty = false;

  // Populate the per-slot WarpParamsUBO (slot 0 in Phase B-2) with the
  // values the pre-recorded command buffer's AffineWarp + Quantize dispatches
  // will read. HOST_COHERENT memory was used at allocation time so the write
  // is immediately visible when the GPU executes the cmd buffer.
  {
    WarpParamsUBO ubo_data = {0};
    ubo_data.a11 = detector->pending_warp_a11;
    ubo_data.a12 = detector->pending_warp_a12;
    ubo_data.a13 = detector->pending_warp_a13;
    ubo_data.a21 = detector->pending_warp_a21;
    ubo_data.a22 = detector->pending_warp_a22;
    ubo_data.a23 = detector->pending_warp_a23;
    ubo_data.fill_value    = detector->pending_warp_fill;
    // Detector never invokes Fproj{Cubic,Bilinear}Y — IMAS pipeline owns those.
    // Populate fproj_bg_value defensively so the slot's UBO stays consistent
    // if a downstream consumer ever reads it.
    ubo_data.fproj_bg_value = 0.0f;
    // AffineWarp output dims (= input_image dims on detect path).
    ubo_data.W_rot         = detector->mem->curr_input_image_width;
    ubo_data.H_rot         = detector->mem->curr_input_image_height;
    ubo_data.H_sub         = detector->mem->curr_input_image_height;
    // Quantize fields.
    ubo_data.canvas_w      = detector->mem->curr_input_image_width;
    ubo_data.canvas_h      = detector->mem->curr_input_image_height;
    ubo_data.valid_w       = detector->quantize_valid_w;
    ubo_data.valid_h       = detector->quantize_valid_h;
    ubo_data.warp_idx      = 0u;
    ubo_data.quantize_fill = detector->quantize_fill_value;
    // sigma_aa / t_factor / gauss_dir_* aren't read by the detector's
    // AffineWarp + Quantize shaders, but populate them defensively so the
    // UBO is in a consistent state.
    ubo_data.sigma_aa      = 0.0f;
    ubo_data.t_factor      = 1.0f;
    ubo_data.gauss_dir_x   = 0.0f;
    ubo_data.gauss_dir_y   = 1.0f;
    memcpy(detector->mem->slots[0].warp_params_ubo_ptr, &ubo_data, sizeof(WarpParamsUBO));
  }

  // Mark the detection pipeline as busy/GPU locked
  vkResetFences(detector->dev->device, 1, &detector->end_of_detection_fence);

  VkPipelineStageFlags wait_dst_transfer_bit_stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkPipelineStageFlags wait_dst_compute_shader_bit_stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;

  VkSubmitInfo submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = NULL};
  if (detector->dev->async_transfer_available)
  {
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = NULL;
    submit_info.pWaitDstStageMask = NULL;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &detector->release_buffer_ownership_command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &detector->buffer_ownership_released_by_transfer_semaphore;
    if (vkQueueSubmit(detector->async_ownership_transfer_queue, 1, &submit_info, NULL) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to submit ownership-release command buffer on async transfer queue");
      return false;
    }
  }

  if (detector->dev->async_transfer_available)
  {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &detector->buffer_ownership_released_by_transfer_semaphore;
    submit_info.pWaitDstStageMask = &wait_dst_compute_shader_bit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &detector->end_of_detection_semaphore;
  }
  else
  {
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = NULL;
    submit_info.pWaitDstStageMask = NULL;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;
  }
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = cmd_buffer_to_submit;
  VkFence detect_submit_fence = detector->dev->async_transfer_available ? NULL : detector->end_of_detection_fence;
  if (vkQueueSubmit(detector->general_queue, 1, &submit_info, detect_submit_fence) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to submit detection command buffer");
    return false;
  }

  if (detector->dev->async_transfer_available)
  {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &detector->end_of_detection_semaphore;
    submit_info.pWaitDstStageMask = &wait_dst_transfer_bit_stage_mask;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &detector->acquire_buffer_ownership_command_buffer;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;
    if (vkQueueSubmit(detector->async_ownership_transfer_queue, 1, &submit_info, detector->end_of_detection_fence) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to submit ownership-release command buffer on async transfer queue");
      return false;
    }
  }
  return true;
}

bool vksift_dispatchSiftDetection(vksift_SiftDetector detector, const uint32_t target_buffer_idx, const bool memory_layout_updated)
{
  return dispatchDetectionCmdBuffer(detector, target_buffer_idx, memory_layout_updated,
                                    &detector->detection_command_buffer);
}

bool vksift_dispatchSiftDetectionFromImas(vksift_SiftDetector detector, const uint32_t target_buffer_idx, const bool memory_layout_updated)
{
  return dispatchDetectionCmdBuffer(detector, target_buffer_idx, memory_layout_updated,
                                    &detector->detection_command_buffer_from_imas);
}

// Phase C-3 helper: fill slot's WarpParamsUBO + SlotDispatchBuffer with the
// per-warp host-side params. No GPU submission; pure host work. Used by both
// the serial entry point (vksift_dispatchFusedImasWarpForSlot) and the
// parallel one (vksift_dispatchParallelIMAS). Float math is single-precision
// throughout to match the shader's f32 input. Non-static — also called from
// vulkansift.c's parallel dispatch entry point.
void vksift_fillFusedWarpState(vksift_SiftDetector detector,
                               uint32_t slot_idx, uint32_t warp_idx,
                               uint32_t W, uint32_t H,
                               float t_factor, float theta_rad,
                               uint32_t canvas_w, uint32_t canvas_h)
{
  // ---- 1. Compute rotated canvas + inverse-rotation affine (same formula
  // as vksift_runImasWarp). ----
  float ca = cosf(theta_rad);
  float sa = sinf(theta_rad);
  int xmin, xmax, ymin, ymax;
  uint32_t W_rot, H_rot;
  vksift_imasComputeRotatedCanvas(W, H, ca, sa, &xmin, &xmax, &ymin, &ymax, &W_rot, &H_rot);

  uint32_t H_t = (t_factor > 1.0f) ? (uint32_t)floorf((float)H_rot / t_factor) : H_rot;
  if (H_t < 1u) H_t = 1u;

  // Inverse rotation matrix (matches vksift_runImasWarp).
  const float a11 = ca;
  const float a12 = -sa;
  const float a13 = ca * (float)xmin - sa * (float)ymin;
  const float a21 = sa;
  const float a22 = ca;
  const float a23 = sa * (float)xmin + ca * (float)ymin;
  const float fill = 0.5f;
  const float sigma_aa = (t_factor > 1.0f) ? 0.8f * sqrtf(t_factor * t_factor - 1.0f) : 0.0f;

  // ---- 2. Populate the slot's WarpParamsUBO ----
  {
    WarpParamsUBO ubo = {0};
    ubo.a11 = a11; ubo.a12 = a12; ubo.a13 = a13;
    ubo.a21 = a21; ubo.a22 = a22; ubo.a23 = a23;
    ubo.fill_value     = fill;
    ubo.fproj_bg_value = 0.0f;
    ubo.W_rot          = W_rot;
    ubo.H_rot          = H_rot;
    ubo.H_sub          = H_t;
    ubo.canvas_w       = canvas_w;
    ubo.canvas_h       = canvas_h;
    ubo.valid_w        = W_rot;
    ubo.valid_h        = H_t;
    ubo.sigma_aa       = sigma_aa;
    ubo.t_factor       = t_factor;
    ubo.quantize_fill  = 0.5f;
    ubo.gauss_dir_x    = 0.0f;
    ubo.gauss_dir_y    = 1.0f;  // IMAS vertical σ_aa blur
    ubo.warp_idx       = warp_idx;

    // ---- Phase D — back-projection params (tilted-frame → input-frame) ----
    // The IMAS forward chain is:
    //   1. AffineWarp (frot):  input → rotated, applying R(-φ).
    //   2. FprojCubic/Bilinear: rotated → tilted, subsampling y by t.
    // So forward = T(1/t) · R(-φ). The inverse (tilted → input) is therefore
    // R(φ) · T(t) — *asymmetric*:
    //   [ cosφ   -t·sinφ ]
    //   [ sinφ    t·cosφ ]
    // The earlier code wrote the *symmetric* shape matrix R(-φ)·T(t)·R(φ)
    // here, which is correct as a CPU-side σ → ellipse conversion (semi-axes
    // 1 and t along the rotated axes) but wrong for back-projecting a
    // *position* — it rotated the back-projected (xi, yi) by an extra φ,
    // putting the boundary check on a curve that doesn't match the actual
    // parallelogram edge. The fix below uses the true asymmetric inverse.
    // σ_max = t is still the largest singular value (unchanged).
    // Asymmetric back-projection R(+θ)·T(t). Matches imas_cpu.jl's
    // tiltedcoor2imagecoor exactly (signs verified at lines 525-529).
    const float cf = cosf(theta_rad);
    const float sf = sinf(theta_rad);
    ubo.bp_a11 = cf;
    ubo.bp_a12 = -t_factor * sf;
    ubo.bp_a21 = sf;
    ubo.bp_a22 =  t_factor * cf;
    ubo.bp_a13 = cf * (float)xmin - sf * (float)ymin;
    ubo.bp_a23 = sf * (float)xmin + cf * (float)ymin;
    ubo.bp_sigma_max  = (t_factor > 1.0f) ? t_factor : 1.0f;
    ubo.bp_boundary_K = 3.0f;
    ubo.bp_input_W    = W;
    ubo.bp_input_H    = H;
    // Derive identity intrinsically from the warp transform, not from
    // warp_idx. Serial callers (vksift_dispatchFusedImasWarpForSlot)
    // pass slot_idx as warp_idx, and parallel callers might pass any
    // permutation of the schedule — making identity depend on warp_idx
    // would silently mis-classify identity for slot>0 in the serial
    // path. The transform itself is unambiguous: t=1, φ=0 is identity.
    ubo.bp_is_identity = ((t_factor == 1.0f) && (theta_rad == 0.0f)) ? 1u : 0u;
    ubo.bp_nb_octaves  = detector->mem->curr_nb_octaves;

    memcpy(detector->mem->slots[slot_idx].warp_params_ubo_ptr, &ubo, sizeof(WarpParamsUBO));
  }

  // ---- 3. Populate the slot's indirect-dispatch buffer ----
  {
    const uint32_t oct0_w = detector->mem->octave_resolutions[0].width;
    const uint32_t oct0_h = detector->mem->octave_resolutions[0].height;
    SlotDispatchBuffer disp = {0};
    disp.affinewarp       = (VkDispatchIndirectCommand){(W_rot + 7u) / 8u, (H_rot + 7u) / 8u, 1u};
    disp.gaussblur        = (VkDispatchIndirectCommand){(W_rot + 7u) / 8u, (H_rot + 7u) / 8u, 1u};
    disp.finvspline_row   = (VkDispatchIndirectCommand){(H_rot + 63u) / 64u, 1u, 1u};
    disp.finvspline_col   = (VkDispatchIndirectCommand){(W_rot + 63u) / 64u, 1u, 1u};
    disp.fproj            = (VkDispatchIndirectCommand){(W_rot + 7u) / 8u, (H_t + 7u) / 8u, 1u};
    disp.quantize         = (VkDispatchIndirectCommand){(canvas_w + 7u) / 8u, (canvas_h + 7u) / 8u, 1u};
    disp.seed_from_input  = (VkDispatchIndirectCommand){(oct0_w + 7u) / 8u, (oct0_h + 7u) / 8u, 1u};
    memcpy(detector->mem->slots[slot_idx].dispatch_buffer_ptr, &disp, sizeof(SlotDispatchBuffer));
  }
}

// Phase B-3 — see sift_detector.h for the contract. Builds the WarpParamsUBO
// + SlotDispatchBuffer for the requested warp, writes them into the slot's
// host-mapped buffers, then submits the pre-recorded fused command buffer for
// the slot. The host populates exactly the fields each shader (IMAS chain +
// Quantize) reads; the SIFT-detect chain has no per-warp params (its canvas
// is curr_input_image_*).
// Phase C-3: ensure the fused-cmd buffers are freshly recorded for the given
// target_buffer_idx (legacy curr_buffer_idx tracking). Returns true if the
// recording is current. Used by both the serial dispatch entry point and the
// parallel one (vulkansift.c::vksift_dispatchParallelIMAS) before submitting
// any of the pre-recorded fused command buffers.
bool vksift_ensureDetectorCmdBuffersRecorded(vksift_SiftDetector detector,
                                             uint32_t target_buffer_idx,
                                             bool memory_layout_updated)
{
  bool need_record = memory_layout_updated || detector->pending_warp_dirty ||
                     detector->curr_buffer_idx != target_buffer_idx ||
                     detector->pending_blur_dirty;
  detector->curr_buffer_idx = target_buffer_idx;
  if (need_record)
  {
    writeDescriptorSets(detector);
    if (!recordCommandBuffers(detector)) return false;
    detector->pending_warp_dirty = false;
    detector->pending_blur_dirty = false;
  }
  return true;
}

bool vksift_dispatchFusedImasWarpForSlot(vksift_SiftDetector detector,
                                         uint32_t slot_idx, const uint32_t target_buffer_idx,
                                         uint32_t W, uint32_t H,
                                         float t_factor, float theta_rad,
                                         uint32_t canvas_w, uint32_t canvas_h,
                                         bool memory_layout_updated)
{
  if (detector == NULL || slot_idx >= VKSIFT_MAX_PYRAMID_SLOTS) return false;

  // ---- 1+2+3. Fill UBO + dispatch buffer for this (slot, warp) ----
  vksift_fillFusedWarpState(detector, slot_idx, slot_idx,
                            W, H, t_factor, theta_rad, canvas_w, canvas_h);

  // ---- 4. Ensure the fused cmd buffer is freshly recorded ----
  if (!vksift_ensureDetectorCmdBuffersRecorded(detector, target_buffer_idx, memory_layout_updated))
  {
    return false;
  }

  // ---- 5. Submit the fused command buffer. Mirrors dispatchDetectionCmdBuffer's
  // non-async-transfer branch (the fused buffer includes its own ownership
  // transfer barriers if async transfer is enabled). ----
  vkResetFences(detector->dev->device, 1, &detector->end_of_detection_fence);

  VkPipelineStageFlags wait_dst_transfer_bit_stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
  VkPipelineStageFlags wait_dst_compute_shader_bit_stage_mask = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  VkSubmitInfo submit_info = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .pNext = NULL};

  if (detector->dev->async_transfer_available)
  {
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &detector->release_buffer_ownership_command_buffer;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &detector->buffer_ownership_released_by_transfer_semaphore;
    if (vkQueueSubmit(detector->async_ownership_transfer_queue, 1, &submit_info, NULL) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to submit ownership-release cmd buf (fused path)");
      return false;
    }

    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &detector->buffer_ownership_released_by_transfer_semaphore;
    submit_info.pWaitDstStageMask = &wait_dst_compute_shader_bit_stage_mask;
    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = &detector->end_of_detection_semaphore;
  }
  else
  {
    submit_info.waitSemaphoreCount = 0;
    submit_info.pWaitSemaphores = NULL;
    submit_info.pWaitDstStageMask = NULL;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;
  }

  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &detector->fused_imas_detect_command_buffer[slot_idx];
  VkFence detect_submit_fence = detector->dev->async_transfer_available
                                 ? NULL : detector->end_of_detection_fence;
  if (vkQueueSubmit(detector->general_queue, 1, &submit_info, detect_submit_fence) != VK_SUCCESS)
  {
    logError(LOG_TAG, "Failed to submit fused IMAS+detect cmd buffer (slot %u)", slot_idx);
    return false;
  }

  if (detector->dev->async_transfer_available)
  {
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = &detector->end_of_detection_semaphore;
    submit_info.pWaitDstStageMask = &wait_dst_transfer_bit_stage_mask;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &detector->acquire_buffer_ownership_command_buffer;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = NULL;
    if (vkQueueSubmit(detector->async_ownership_transfer_queue, 1, &submit_info,
                      detector->end_of_detection_fence) != VK_SUCCESS)
    {
      logError(LOG_TAG, "Failed to submit ownership-acquire cmd buf (fused path)");
      return false;
    }
  }
  return true;
}

void vksift_setPendingAffineWarp(vksift_SiftDetector detector,
                                 float a11, float a12, float a13,
                                 float a21, float a22, float a23,
                                 float fill_value)
{
  if (detector == NULL) return;
  detector->pending_warp_a11 = a11;
  detector->pending_warp_a12 = a12;
  detector->pending_warp_a13 = a13;
  detector->pending_warp_a21 = a21;
  detector->pending_warp_a22 = a22;
  detector->pending_warp_a23 = a23;
  detector->pending_warp_fill = fill_value;
  detector->pending_warp_dirty = true;
}

void vksift_setPendingPreBlur(vksift_SiftDetector detector,
                              float sigma, float dir_x, float dir_y)
{
  if (detector == NULL) return;
  detector->pending_blur_sigma = sigma;
  detector->pending_blur_dir_x = dir_x;
  detector->pending_blur_dir_y = dir_y;
  detector->pending_blur_dirty = true;
}

void vksift_destroySiftDetector(vksift_SiftDetector *detector_ptr)
{
  assert(detector_ptr != NULL);
  assert(*detector_ptr != NULL); // vksift_destroySiftDetector shouldn't be called on NULL vksift_SiftMemory object
  vksift_SiftDetector detector = *detector_ptr;

  // Destroy sampler
  VK_NULL_SAFE_DELETE(detector->image_sampler, vkDestroySampler(detector->dev->device, detector->image_sampler, NULL));

  // Destroy sync objects
  VK_NULL_SAFE_DELETE(detector->end_of_detection_semaphore, vkDestroySemaphore(detector->dev->device, detector->end_of_detection_semaphore, NULL));
  VK_NULL_SAFE_DELETE(detector->end_of_detection_fence, vkDestroyFence(detector->dev->device, detector->end_of_detection_fence, NULL));
  // Phase C-async: optional second fence (compute-queue half of parallel-IMAS).
  VK_NULL_SAFE_DELETE(detector->end_of_detection_fence_compute,
                      vkDestroyFence(detector->dev->device, detector->end_of_detection_fence_compute, NULL));
  // Phase E: optional graphics→compute sync semaphore.
  VK_NULL_SAFE_DELETE(detector->parallel_compute_start_semaphore,
                      vkDestroySemaphore(detector->dev->device, detector->parallel_compute_start_semaphore, NULL));
  if (detector->dev->async_transfer_available)
  {
    VK_NULL_SAFE_DELETE(detector->buffer_ownership_released_by_transfer_semaphore,
                        vkDestroySemaphore(detector->dev->device, detector->buffer_ownership_released_by_transfer_semaphore, NULL));
  }

  // Destroy command pools
  VK_NULL_SAFE_DELETE(detector->general_command_pool, vkDestroyCommandPool(detector->dev->device, detector->general_command_pool, NULL));
  if (detector->dev->async_transfer_available)
  {
    VK_NULL_SAFE_DELETE(detector->async_transfer_command_pool, vkDestroyCommandPool(detector->dev->device, detector->async_transfer_command_pool, NULL));
  }
  // Phase C-async: optional async-compute pool (NULL if no dedicated compute queue).
  // Destroying the pool frees the per-slot cmd buffers in
  // fused_imas_detect_command_buffer_compute[] implicitly.
  if (detector->dev->async_compute_available)
  {
    VK_NULL_SAFE_DELETE(detector->async_compute_command_pool, vkDestroyCommandPool(detector->dev->device, detector->async_compute_command_pool, NULL));
  }

  // Destroy pipelines and resource bindings
  // PreBlur1D (ASIFT σ_aa pre-blur)
  VK_NULL_SAFE_DELETE(detector->preblur_pipeline, vkDestroyPipeline(detector->dev->device, detector->preblur_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->preblur_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->preblur_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->preblur_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->preblur_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->preblur_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->preblur_desc_set_layout, NULL));
  // Affine warp (ASIFT batch path)
  VK_NULL_SAFE_DELETE(detector->affinewarp_pipeline, vkDestroyPipeline(detector->dev->device, detector->affinewarp_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->affinewarp_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->affinewarp_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->affinewarp_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->affinewarp_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->affinewarp_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->affinewarp_desc_set_layout, NULL));
  // Shared WarpParamsUBO descriptor pool + layout
  VK_NULL_SAFE_DELETE(detector->warp_ubo_desc_pool,
                      vkDestroyDescriptorPool(detector->dev->device, detector->warp_ubo_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->warp_ubo_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->warp_ubo_desc_set_layout, NULL));
  // Gaussian blur
  VK_NULL_SAFE_DELETE(detector->blur_pipeline, vkDestroyPipeline(detector->dev->device, detector->blur_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->blur_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->blur_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->blur_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->blur_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->blur_desc_set_layout, vkDestroyDescriptorSetLayout(detector->dev->device, detector->blur_desc_set_layout, NULL));
  // Difference of Gaussian
  VK_NULL_SAFE_DELETE(detector->dog_pipeline, vkDestroyPipeline(detector->dev->device, detector->dog_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->dog_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->dog_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->dog_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->dog_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->dog_desc_set_layout, vkDestroyDescriptorSetLayout(detector->dev->device, detector->dog_desc_set_layout, NULL));
  // Downsample2x (Lowe pixel-aligned octave downsample)
  VK_NULL_SAFE_DELETE(detector->downsample_pipeline, vkDestroyPipeline(detector->dev->device, detector->downsample_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->downsample_pipeline_layout,
                      vkDestroyPipelineLayout(detector->dev->device, detector->downsample_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->downsample_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->downsample_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->downsample_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->downsample_desc_set_layout, NULL));
  // Upsample2xLinear (octave-0 2× linear upsample, replaces vkCmdBlitImage)
  VK_NULL_SAFE_DELETE(detector->upsample_pipeline, vkDestroyPipeline(detector->dev->device, detector->upsample_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->upsample_pipeline_layout,
                      vkDestroyPipelineLayout(detector->dev->device, detector->upsample_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->upsample_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->upsample_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->upsample_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->upsample_desc_set_layout, NULL));
  // SiftSeedFromInput (IMAS-fused octave-0 fast path)
  VK_NULL_SAFE_DELETE(detector->seed_from_input_pipeline, vkDestroyPipeline(detector->dev->device, detector->seed_from_input_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->seed_from_input_pipeline_layout,
                      vkDestroyPipelineLayout(detector->dev->device, detector->seed_from_input_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->seed_from_input_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->seed_from_input_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->seed_from_input_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->seed_from_input_desc_set_layout, NULL));
  // Extract keypoints
  VK_NULL_SAFE_DELETE(detector->extractkpts_pipeline, vkDestroyPipeline(detector->dev->device, detector->extractkpts_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->extractkpts_2d_pipeline, vkDestroyPipeline(detector->dev->device, detector->extractkpts_2d_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->extractkpts_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->extractkpts_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->extractkpts_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->extractkpts_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->extractkpts_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->extractkpts_desc_set_layout, NULL));
  // Phase D — BackProjectFeatures pipeline + descriptor sets
  VK_NULL_SAFE_DELETE(detector->backproject_pipeline, vkDestroyPipeline(detector->dev->device, detector->backproject_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->backproject_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->backproject_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->backproject_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->backproject_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->backproject_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->backproject_desc_set_layout, NULL));
  // Compute orientation
  VK_NULL_SAFE_DELETE(detector->orientation_pipeline, vkDestroyPipeline(detector->dev->device, detector->orientation_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->orientation_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->orientation_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->orientation_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->orientation_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->orientation_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->orientation_desc_set_layout, NULL));
  // Compute descriptor
  VK_NULL_SAFE_DELETE(detector->descriptor_pipeline, vkDestroyPipeline(detector->dev->device, detector->descriptor_pipeline, NULL));
  VK_NULL_SAFE_DELETE(detector->descriptor_pipeline_layout, vkDestroyPipelineLayout(detector->dev->device, detector->descriptor_pipeline_layout, NULL));
  VK_NULL_SAFE_DELETE(detector->descriptor_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->descriptor_desc_pool, NULL));
  VK_NULL_SAFE_DELETE(detector->descriptor_desc_set_layout,
                      vkDestroyDescriptorSetLayout(detector->dev->device, detector->descriptor_desc_set_layout, NULL));
  // RGBA→Gray conversion
  if (detector->use_rgba_input)
  {
    VK_NULL_SAFE_DELETE(detector->rgba_convert_pipeline, vkDestroyPipeline(detector->dev->device, detector->rgba_convert_pipeline, NULL));
    VK_NULL_SAFE_DELETE(detector->rgba_convert_pipeline_layout,
                        vkDestroyPipelineLayout(detector->dev->device, detector->rgba_convert_pipeline_layout, NULL));
    VK_NULL_SAFE_DELETE(detector->rgba_convert_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->rgba_convert_desc_pool, NULL));
    VK_NULL_SAFE_DELETE(detector->rgba_convert_desc_set_layout,
                        vkDestroyDescriptorSetLayout(detector->dev->device, detector->rgba_convert_desc_set_layout, NULL));
  }
  // RGB→Gray conversion
  if (detector->use_rgb_input)
  {
    VK_NULL_SAFE_DELETE(detector->rgb_convert_pipeline, vkDestroyPipeline(detector->dev->device, detector->rgb_convert_pipeline, NULL));
    VK_NULL_SAFE_DELETE(detector->rgb_convert_pipeline_layout,
                        vkDestroyPipelineLayout(detector->dev->device, detector->rgb_convert_pipeline_layout, NULL));
    VK_NULL_SAFE_DELETE(detector->rgb_convert_desc_pool, vkDestroyDescriptorPool(detector->dev->device, detector->rgb_convert_desc_pool, NULL));
    VK_NULL_SAFE_DELETE(detector->rgb_convert_desc_set_layout,
                        vkDestroyDescriptorSetLayout(detector->dev->device, detector->rgb_convert_desc_set_layout, NULL));
  }

  // Per-slot shader-region timestamp query pools.
  if (detector->shader_timestamp_pools != NULL)
  {
    for (uint32_t s = 0u; s < detector->mem->nb_pyramid_slots; ++s)
    {
      VK_NULL_SAFE_DELETE(detector->shader_timestamp_pools[s],
                          vkDestroyQueryPool(detector->dev->device, detector->shader_timestamp_pools[s], NULL));
    }
    free(detector->shader_timestamp_pools);
    detector->shader_timestamp_pools = NULL;
  }

  // Free descriptor arrays
  VK_NULL_SAFE_DELETE(detector->gaussian_kernels, free(detector->gaussian_kernels));
  VK_NULL_SAFE_DELETE(detector->gaussian_kernel_sizes, free(detector->gaussian_kernel_sizes));
  VK_NULL_SAFE_DELETE(detector->blur_desc_sets, free(detector->blur_desc_sets));
  VK_NULL_SAFE_DELETE(detector->dog_desc_sets, free(detector->dog_desc_sets));
  VK_NULL_SAFE_DELETE(detector->downsample_desc_sets, free(detector->downsample_desc_sets));
  VK_NULL_SAFE_DELETE(detector->extractkpts_desc_sets, free(detector->extractkpts_desc_sets));
  VK_NULL_SAFE_DELETE(detector->backproject_desc_sets, free(detector->backproject_desc_sets));
  VK_NULL_SAFE_DELETE(detector->orientation_desc_sets, free(detector->orientation_desc_sets));
  VK_NULL_SAFE_DELETE(detector->descriptor_desc_sets, free(detector->descriptor_desc_sets));

  // Release vksift_SiftDetector memory
  free(*detector_ptr);
  *detector_ptr = NULL;
}