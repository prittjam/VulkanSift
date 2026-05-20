#include "vulkansift/vulkansift.h"

// vkenv
#include "vulkansift/vkenv/debug_presenter.h"
#include "vulkansift/vkenv/logger.h"
#include "vulkansift/vkenv/vulkan_device.h"
#include "vulkansift/vkenv/vulkan_surface.h"
#include "vulkansift/vkenv/vulkan_utils.h"

// vksift
#include "vulkansift/sift_detector.h"
#include "vulkansift/sift_imas.h"
#include "vulkansift/sift_matcher.h"
#include "vulkansift/sift_memory.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static const char LOG_TAG[] = "VulkanSift";
static bool swapchain_extensions_supported;

// Input validity checks functions
static bool checkConfigCond(bool cond, const char *msg_on_cond_false);
static bool isConfigurationValid(const vksift_Config *config);
static bool isBufferIdxValid(vksift_Instance instance, const uint32_t buffer_idx);
static bool isInputResolutionValid(vksift_Instance instance, const uint32_t input_width, const uint32_t input_height);
static bool isInputFeatureCoundValid(vksift_Instance instance, const uint32_t nb_feats);
static bool isInputOctaveIdxValid(vksift_Instance instance, const uint32_t octave_idx);
static bool isInputScaleIdxValid(vksift_Instance instance, const uint32_t scale_idx, bool is_dog);

static void default_error_callback(vksift_Result err_type)
{
  switch (err_type)
  {
  case VKSIFT_INVALID_INPUT_ERROR:
    logDebug(LOG_TAG, "Aborting after invalid input error...");
    break;
  case VKSIFT_VULKAN_ERROR:
    logDebug(LOG_TAG, "Aborting after Vulkan error...");
    break;
  default:
    break;
  }
  abort();
}

static vksift_Config vksift_Config_Default = {.input_image_max_size = 1920u * 1080u,
                                              .input_image_max_width = 0u,   // 0 = use legacy square ceiling
                                              .input_image_max_height = 0u,
                                              .sift_buffer_count = 2u, // minimum number of buffer to support the feature matching function
                                              .max_nb_sift_per_buffer = 100000u,
                                              .use_input_upsampling = true, // provide the best results (higher processing time)
                                              .nb_octaves = 0,              // defined by implementation
                                              .nb_scales_per_octave = 3u,   // Lowe's paper
                                              .input_image_blur_level = 0.5f,
                                              .seed_scale_sigma = 1.6f, // Lowe's paper
                                              .intensity_threshold = 0.04f,
                                              .edge_threshold = 10.f,               // Lowe's paper
                                              .max_nb_orientation_per_keypoint = 4, // no more than 4 descriptor for a single keypoint position
                                              .descriptor_format = VKSIFT_DESCRIPTOR_FORMAT_UBC, // compatibility with OpenCV and SiftGPU
                                              .gpu_device_index = -1,                            // GPU auto-selection
                                              .use_hardware_interpolated_blur = true,            // faster with no noticeable quality loss
                                              .pyramid_precision_mode = VKSIFT_PYRAMID_PRECISION_FLOAT32,
                                              .on_error_callback_function = default_error_callback,
                                              .detection_only = false,
                                              .use_2d_nms = false,
                                              .use_rgba_input = false,
                                              .use_rgb_input = false,
                                              .use_gpu_debug_functions = false,
                                              .gpu_debug_external_window_info = {.context = NULL, .window = NULL},
                                              .nb_pyramid_slots = 1u};

vksift_Config vksift_getDefaultConfig() { return vksift_Config_Default; }

vksift_Result vksift_loadVulkan()
{
  vkenv_InstanceConfig instance_config = {.application_name = "VulkanSift",
                                          .application_version = VK_MAKE_VERSION(1, 0, 0),
                                          .engine_name = "",
                                          .engine_version = VK_MAKE_VERSION(1, 0, 0),
                                          .validation_layer_count = 0,
                                          .validation_layers = NULL,
                                          .instance_extension_count = 0,
                                          .instance_extensions = NULL};

#ifndef NDEBUG
  // Activate Vulkan debug layers on debug mode only
  const char *validation_layer_name = "VK_LAYER_KHRONOS_validation";
  instance_config.validation_layer_count = 1;
  instance_config.validation_layers = &validation_layer_name;
#endif

  // Try to create an instance with the rendering extensions and the debug marker extension
  const char *instance_extensions[3];
  instance_extensions[0] = (const char *)VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
  instance_extensions[1] = (const char *)VK_KHR_SURFACE_EXTENSION_NAME;
  instance_extensions[2] = vkenv_getSurfaceExtensionName();
  instance_config.instance_extension_count = 3;
  instance_config.instance_extensions = instance_extensions;

  swapchain_extensions_supported = true;
  if (!vkenv_createInstance(&instance_config))
  {
    swapchain_extensions_supported = false;
    logWarning(LOG_TAG, "Could not initialize Vulkan instance with swapchain extensions. Trying without any extensions...");
    // If instance creation fails, try without any instance extensions (GPU debug won't be supported)
    instance_config.instance_extension_count = 0;
    instance_config.instance_extensions = NULL;
    if (!vkenv_createInstance(&instance_config))
    {
      logError(LOG_TAG, "vksift_loadVulkan() failure when seting up the Vulkan instance.");
      return VKSIFT_VULKAN_ERROR;
    }
  }
  logInfo(LOG_TAG, "vksift_loadVulkan() success");
  return VKSIFT_SUCCESS;
}

void vksift_unloadVulkan() { vkenv_destroyInstance(); }

void vksift_getAvailableGPUs(uint32_t *gpu_count, VKSIFT_GPU_NAME *gpu_names)
{
  if (gpu_names == NULL)
  {
    vkenv_getPhysicalDevicesProperties(gpu_count, NULL);
  }
  else
  {
    VkPhysicalDeviceProperties *devices_props = (VkPhysicalDeviceProperties *)malloc(sizeof(VkPhysicalDeviceProperties) * (*gpu_count));
    vkenv_getPhysicalDevicesProperties(gpu_count, devices_props);
    for (uint32_t i = 0; i < *gpu_count; i++)
    {
      memcpy(gpu_names[i], devices_props[i].deviceName, sizeof(VKSIFT_GPU_NAME));
    }
    free(devices_props);
  }
}

void vksift_setLogLevel(vksift_LogLevel level)
{
  switch (level)
  {
  case VKSIFT_NO_LOG:
    vkenv_setLogLevel(VKENV_LOG_NONE);
    break;
  case VKSIFT_LOG_ERROR:
    vkenv_setLogLevel(VKENV_LOG_ERROR);
    break;
  case VKSIFT_LOG_WARNING:
    vkenv_setLogLevel(VKENV_LOG_WARNING);
    break;
  case VKSIFT_LOG_INFO:
    vkenv_setLogLevel(VKENV_LOG_INFO);
    break;
  case VKSIFT_LOG_DEBUG:
    vkenv_setLogLevel(VKENV_LOG_DEBUG);
    break;
  default:
    logError(LOG_TAG, "vksift_LogLevel in vksift_setLogLevel() is not handled");
    break;
  }
}

typedef struct vksift_Instance_T
{
  vkenv_Device vulkan_device;
  vksift_SiftMemory sift_memory;
  vksift_SiftDetector sift_detector;
  vksift_SiftMatcher sift_matcher;
  vkenv_DebugPresenter debug_presenter; // NULL if vksift_ExternalWindowInfo is not provided

  // Lazy-initialised IMAS (ASIFT) pipeline. NULL until first vksift_runImas
  // call. Owns its own Vulkan objects + readback buffer; samples from the
  // shared input_image and writes the final tilted result to a host-mapped
  // buffer accessible via vksift_getImasReadbackPtr().
  vksift_ImasPipeline imas_pipeline;

  void (*error_cb_func)(vksift_Result);
} vksift_Instance_T;

vksift_Result vksift_createInstance(vksift_Instance *instance_ptr, const vksift_Config *config)
{
  assert(instance_ptr != NULL);
  assert(*instance_ptr == NULL);
  assert(config != NULL);

  // Check that the Vulkan instance is available
  if (vkenv_getInstance() == NULL)
  {
    logError(LOG_TAG, "vksift_createInstance() failure: Vulkan API not available. vksift_loadVulkan() must be called before using this function.");
    return VKSIFT_VULKAN_ERROR;
  }

  // Check configuration validity
  if (!isConfigurationValid(config))
  {
    logError(LOG_TAG, "vksift_createInstance() failure: Invalid configuration detected.");
    return VKSIFT_INVALID_INPUT_ERROR;
  }

  *instance_ptr = (vksift_Instance)malloc(sizeof(vksift_Instance_T));
  vksift_Instance instance = *instance_ptr;
  memset(instance, 0, sizeof(vksift_Instance_T));

  instance->error_cb_func = config->on_error_callback_function;

  // Setup device
  // We need two async transfer queue to properly do async transfers, one is only used by the memory for GPU download/upload, the other is for
  // detection/matching SIFT buffer ownership transfers
  vkenv_DeviceConfig gpu_config = {.device_extension_count = 0,
                                   .device_extensions = NULL,
                                   .nb_general_queues = 1,
                                   .nb_async_compute_queues = 0,
                                   .nb_async_transfer_queues = 2,
                                   .target_device_idx = config->gpu_device_index};

  const char *device_extension_name = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
  if (swapchain_extensions_supported)
  {
    gpu_config.device_extension_count = 1;
    gpu_config.device_extensions = &device_extension_name;
  }

  if (!vkenv_createDevice(&instance->vulkan_device, &gpu_config))
  {
    logError(LOG_TAG, "vksift_createInstance() failure: An error occured when creating the Vulkan device");
    vksift_destroyInstance(instance_ptr);
    return VKSIFT_VULKAN_ERROR;
  }

  if (!vksift_createSiftMemory(instance->vulkan_device, &instance->sift_memory, config))
  {
    logError(LOG_TAG, "vksift_createInstance() failure: Failed to setup the required memory objects");
    vksift_destroyInstance(instance_ptr);
    return VKSIFT_VULKAN_ERROR;
  }

  if (!vksift_createSiftDetector(instance->vulkan_device, instance->sift_memory, &instance->sift_detector, config))
  {
    logError(LOG_TAG, "vksift_createInstance() failure: Failed to setup the SIFT detector");
    vksift_destroyInstance(instance_ptr);
    return VKSIFT_VULKAN_ERROR;
  }

  if (!vksift_createSiftMatcher(instance->vulkan_device, instance->sift_memory, &instance->sift_matcher))
  {
    logError(LOG_TAG, "vksift_createInstance() failure: Failed to setup the SIFT matcher");
    vksift_destroyInstance(instance_ptr);
    return VKSIFT_VULKAN_ERROR;
  }

  if (config->use_gpu_debug_functions)
  {
    if (swapchain_extensions_supported == false)
    {
      logError(LOG_TAG, "vksift_createInstance() failure: external window information specified but Vulkan instance doesn't support rendering.");
      vksift_destroyInstance(instance_ptr);
      return VKSIFT_VULKAN_ERROR;
    }

    // Setup the DebugPresenter from the external window informations
    vkenv_ExternalWindowInfo window_info = {.context = config->gpu_debug_external_window_info.context,
                                            .window = config->gpu_debug_external_window_info.window};
    if (!vkenv_createDebugPresenter(instance->vulkan_device, &instance->debug_presenter, &window_info))
    {
      logError(LOG_TAG, "vksift_createInstance() failure: An error occured when preparing the debug window");
      vksift_destroyInstance(instance_ptr);
      return VKSIFT_VULKAN_ERROR;
    }
  }

  logInfo(LOG_TAG, "vksift_createInstance() success");
  return VKSIFT_SUCCESS;
}

void vksift_destroyInstance(vksift_Instance *instance_ptr)
{
  assert(instance_ptr != NULL);
  assert(*instance_ptr != NULL); // vksift_destroyInstance shouldn't be called on NULL Instance
  vksift_Instance instance = *instance_ptr;

  if (instance->vulkan_device != NULL)
  {
    // Wait for anything running on the GPU to finish
    vkDeviceWaitIdle(instance->vulkan_device->device);
  }

  // Destroy IMAS pipeline (if lazy-created)
  if (instance->imas_pipeline != NULL)
  {
    vksift_destroyImasPipeline(&instance->imas_pipeline);
  }

  // Destroy SiftMatcher
  VK_NULL_SAFE_DELETE(instance->sift_matcher, vksift_destroySiftMatcher(&instance->sift_matcher));

  // Destroy SiftDetector
  VK_NULL_SAFE_DELETE(instance->sift_detector, vksift_destroySiftDetector(&instance->sift_detector));

  // Destroy SiftMemory
  VK_NULL_SAFE_DELETE(instance->sift_memory, vksift_destroySiftMemory(&instance->sift_memory));

  // Destroy DebugPresenter
  VK_NULL_SAFE_DELETE(instance->debug_presenter, vkenv_destroyDebugPresenter(instance->vulkan_device, &instance->debug_presenter));

  // Destroy Vulkan device
  VK_NULL_SAFE_DELETE(instance->vulkan_device, vkenv_destroyDevice(&instance->vulkan_device));

  // Releave vksift_Instance memory
  free(*instance_ptr);
  *instance_ptr = NULL;
}

bool vksift_isBufferAvailable(vksift_Instance instance, const uint32_t gpu_buffer_id)
{
  if (vkGetFenceStatus(instance->vulkan_device->device, instance->sift_detector->end_of_detection_fence) == VK_NOT_READY &&
      gpu_buffer_id == instance->sift_detector->curr_buffer_idx)
  {
    // Detection is running and will fill the buffer
    return false;
  }
  else if (vkGetFenceStatus(instance->vulkan_device->device, instance->sift_matcher->end_of_matching_fence) == VK_NOT_READY &&
           (gpu_buffer_id == instance->sift_matcher->curr_buffer_A_idx || gpu_buffer_id == instance->sift_matcher->curr_buffer_B_idx))
  {
    // Matcher is running and uses the buffer
    return false;
  }
  else
  {
    return true;
  }
}

void vksift_setPendingAffineWarpInstance(vksift_Instance instance,
                                          float a11, float a12, float a13,
                                          float a21, float a22, float a23,
                                          float fill_value)
{
  if (instance == NULL || instance->sift_detector == NULL) return;
  vksift_setPendingAffineWarp(instance->sift_detector, a11, a12, a13, a21, a22, a23, fill_value);
}

void vksift_setPendingPreBlurInstance(vksift_Instance instance,
                                       float sigma, float dir_x, float dir_y)
{
  if (instance == NULL || instance->sift_detector == NULL) return;
  vksift_setPendingPreBlur(instance->sift_detector, sigma, dir_x, dir_y);
}

void vksift_detectFeatures(vksift_Instance instance, const uint8_t *image_data, const uint32_t image_width, const uint32_t image_height,
                           const uint32_t gpu_buffer_id)
{
  if (!isBufferIdxValid(instance, gpu_buffer_id) || !isInputResolutionValid(instance, image_width, image_height))
  {
    logError(LOG_TAG, "vksift_detectFeatures() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // If a detection or matching pipeline is running, we wait for it to end
  VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);

  // Prepare memory for input resolution and update target buffer structure
  bool memory_layout_updated = false;
  if (!vksift_prepareSiftMemoryForDetection(instance->sift_memory, image_data, image_width, image_height, gpu_buffer_id, &memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_detectFeatures() error: Failed to prepare the SiftMemory instance for the input image and target buffer");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
    return;
  }

  // Run the sift detector algorithm
  if (!vksift_dispatchSiftDetection(instance->sift_detector, gpu_buffer_id, memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_detectFeatures() error: Failed to start the detection pipeline.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

uint32_t vksift_getFeaturesNumber(vksift_Instance instance, const uint32_t gpu_buffer_id)
{
  if (!isBufferIdxValid(instance, gpu_buffer_id))
  {
    logError(LOG_TAG, "vksift_getFeaturesNumber() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return 0;
  }

  // If a GPU task is currently using the buffer, wait for it to be available
  if (!vksift_isBufferAvailable(instance, gpu_buffer_id))
  {
    VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
    vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);
  }

  uint32_t feat_count = 0;
  if (!vksift_Memory_getBufferFeatureCount(instance->sift_memory, gpu_buffer_id, &feat_count))
  {
    logError(LOG_TAG, "vksift_getFeaturesNumber() error when retrieving the number of detected SIFT features.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
  return feat_count;
}

void vksift_downloadFeatures(vksift_Instance instance, vksift_Feature *feats_ptr, uint32_t gpu_buffer_id)
{
  if (!isBufferIdxValid(instance, gpu_buffer_id))
  {
    logError(LOG_TAG, "vksift_downloadFeatures() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // If a GPU task is currently using the buffer, wait for it to be available
  if (!vksift_isBufferAvailable(instance, gpu_buffer_id))
  {
    VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
    vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);
  }

  if (!vksift_Memory_copyBufferFeaturesFromGPU(instance->sift_memory, gpu_buffer_id, feats_ptr))
  {
    logError(LOG_TAG, "vksift_downloadFeatures() error when downloading detection results.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

void vksift_uploadFeatures(vksift_Instance instance, const vksift_Feature *feats_ptr, const uint32_t nb_feats, const uint32_t gpu_buffer_id)
{
  if (!isBufferIdxValid(instance, gpu_buffer_id) || !isInputFeatureCoundValid(instance, nb_feats))
  {
    logError(LOG_TAG, "vksift_uploadFeatures() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // If a GPU task is currently using the buffer, wait for it to be available
  if (!vksift_isBufferAvailable(instance, gpu_buffer_id))
  {
    VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
    vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);
  }

  if (!vksift_Memory_copyBufferFeaturesToGPU(instance->sift_memory, gpu_buffer_id, feats_ptr, nb_feats))
  {
    logError(LOG_TAG, "vksift_uploadFeatures() error when uploading SIFT features to GPU memory.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

void vksift_matchFeatures(vksift_Instance instance, uint32_t gpu_buffer_id_A, uint32_t gpu_buffer_id_B)
{
  if (!isBufferIdxValid(instance, gpu_buffer_id_A) || !isBufferIdxValid(instance, gpu_buffer_id_B))
  {
    logError(LOG_TAG, "vksift_matchFeatures() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // If a detection or matching pipeline is running, we wait for it to end
  VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);

  if (!vksift_prepareSiftMemoryForMatching(instance->sift_memory, gpu_buffer_id_A, gpu_buffer_id_B))
  {
    logError(LOG_TAG, "vksift_matchFeatures() error: Failed to prepare the SIFT buffers for the matching pipeline.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }

  if (!vksift_dispatchSiftMatching(instance->sift_matcher, gpu_buffer_id_A, gpu_buffer_id_B))
  {
    logError(LOG_TAG, "vksift_matchFeatures() error: Failed to start the matching pipeline.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

uint32_t vksift_getMatchesNumber(vksift_Instance instance)
{
  uint32_t nb_matches = 0u;
  vksift_Memory_getBufferMatchesCount(instance->sift_memory, &nb_matches);
  return nb_matches;
}

void vksift_downloadMatches(vksift_Instance instance, vksift_Match_2NN *matches)
{
  // If a GPU matching pipeline is currently using the matches buffer, wait for the pipeline to end
  VkFence fences[1] = {instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(instance->vulkan_device->device, 1, fences, VK_TRUE, UINT64_MAX);

  if (!vksift_Memory_copyBufferMatchesFromGPU(instance->sift_memory, matches))
  {
    logError(LOG_TAG, "vksift_downloadMatches() error when downloading SIFT matches from GPU memory.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

///////////////////////////////////////////////////////////////////////
// Scale-space access functions (for debug and visualization)
uint8_t vksift_getScaleSpaceNbOctaves(vksift_Instance instance) { return instance->sift_memory->curr_nb_octaves; }

void vksift_getScaleSpaceOctaveResolution(vksift_Instance instance, const uint8_t octave, uint32_t *octave_images_width, uint32_t *octave_images_height)
{
  if (octave > instance->sift_memory->curr_nb_octaves)
  {
    logError(LOG_TAG, "vksift_getScaleSpaceOctaveResolution() error: invalid input. Requested octave idx is %d but the current number of octave is %d",
             octave, instance->sift_memory->curr_nb_octaves);
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }
  *octave_images_width = instance->sift_memory->octave_resolutions[octave].width;
  *octave_images_height = instance->sift_memory->octave_resolutions[octave].height;
}

void vksift_downloadScaleSpaceImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *blurred_image)
{
  if (!isInputOctaveIdxValid(instance, octave) || !isInputScaleIdxValid(instance, scale, false))
  {
    logError(LOG_TAG, "vksift_downloadScaleSpaceImage() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // Images cannot be transferred when a detection is running, wait for the fence to be sure this is not the case
  VkFence fences[1] = {instance->sift_detector->end_of_detection_fence};
  vkWaitForFences(instance->vulkan_device->device, 1, fences, VK_TRUE, UINT64_MAX);

  if (!vksift_Memory_copyPyramidImageFromGPU(instance->sift_memory, octave, scale, false, blurred_image))
  {
    logError(LOG_TAG, "vksift_downloadScaleSpaceImage() error when downloading pyramid blurred image from GPU memory.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

void vksift_downloadDoGImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *dog_image)
{
  if (!isInputOctaveIdxValid(instance, octave) || !isInputScaleIdxValid(instance, scale, true))
  {
    logError(LOG_TAG, "vksift_downloadDoGImage() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  // Images cannot be transferred when a detection is running, wait for the fence to be sure this is not the case
  VkFence fences[1] = {instance->sift_detector->end_of_detection_fence};
  vkWaitForFences(instance->vulkan_device->device, 1, fences, VK_TRUE, UINT64_MAX);

  if (!vksift_Memory_copyPyramidImageFromGPU(instance->sift_memory, octave, scale, true, dog_image))
  {
    logError(LOG_TAG, "vksift_downloadDoGImage() error when downloading pyramid DoG image from GPU memory.");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}
///////////////////////////////////////////////////////////////////////

void vksift_presentDebugFrame(vksift_Instance instance)
{
  if (instance->debug_presenter != NULL)
  {
    if (!vkenv_presentDebugFrame(instance->vulkan_device, instance->debug_presenter))
    {
      logError(LOG_TAG, "vksift_presentDebugFrame(): error when rendering a debug frame to the provided window.");
      instance->error_cb_func(VKSIFT_VULKAN_ERROR);
      return;
    }
  }
  else
  {
    logWarning(LOG_TAG, "vksift_presentDebugFrame() was called but instance has no external window configured.");
  }
}

///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////
// Input validity checking functions
static bool checkConfigCond(bool cond, const char *msg_on_cond_false)
{
  if (!cond)
  {
    logError(LOG_TAG, msg_on_cond_false);
  }
  return cond;
}

static bool isConfigurationValid(const vksift_Config *config)
{
  // Check that the gaussian kernel sigma for the scale-space seed image if superior or equals to 0.
  bool valid_seed_gaussian_kernel = ((config->use_input_upsampling ? 2.f : 1.f) * config->input_image_blur_level) <= config->seed_scale_sigma;

  bool config_valid = true;
  config_valid &= checkConfigCond(config->input_image_max_size >= 1024, "Invalid configuration: input image size must be greater than or equal to 1024");
  config_valid &= checkConfigCond(config->sift_buffer_count > 0, "Invalid configuration: number of SIFT buffers must be greater than zero");
  config_valid &=
      checkConfigCond(config->max_nb_sift_per_buffer > 0, "Invalid configuration: number of SIFT features per buffers must be greater than zero");
  config_valid &= checkConfigCond(config->nb_scales_per_octave > 0, "Invalid configuration: number of scales per octave must be greater than zero");
  config_valid &= checkConfigCond(config->input_image_blur_level >= 0.f, "Invalid configuration: input image blur level cannot be negative");
  config_valid &= checkConfigCond(config->seed_scale_sigma >= 0, "Invalid configuration: seed scale blur level cannot be negative");
  config_valid &=
      checkConfigCond(valid_seed_gaussian_kernel,
                      "Invalid configuration: the input image blur level (2x if upscaling activated) must be less than the seed scale blur level");
  config_valid &= checkConfigCond(config->intensity_threshold >= 0.f, "Invalid configuration: the DoG intensity threshold cannot be negative");
  config_valid &= checkConfigCond(config->edge_threshold >= 0.f, "Invalid configuration: the DoG edge threshold cannot be negative");
  config_valid &= checkConfigCond(config->edge_threshold >= 0.f, "Invalid configuration: the DoG edge threshold cannot be negative");
  config_valid &= checkConfigCond(config->on_error_callback_function != NULL, "Invalid configuration: the error callback function must not be NULL");

  switch (config->pyramid_precision_mode)
  {
  case VKSIFT_PYRAMID_PRECISION_FLOAT16:
    break;
  case VKSIFT_PYRAMID_PRECISION_FLOAT32:
    break;
  default:
    logError(LOG_TAG, "Invalid configuration: invalid scale-space pyramid format precision specified)");
    config_valid &= false;
    break;
  }

  return config_valid;
}

static bool isBufferIdxValid(vksift_Instance instance, const uint32_t buffer_idx)
{
  if (buffer_idx > instance->sift_memory->nb_sift_buffer)
  {
    logError(LOG_TAG, "Provided target buffer index is (%d) but the number of reserved buffers is (%d).", buffer_idx,
             instance->sift_memory->nb_sift_buffer);
    return false;
  }
  else
  {
    return true;
  }
}

static bool isInputResolutionValid(vksift_Instance instance, const uint32_t input_width, const uint32_t input_height)
{
  uint32_t input_size = input_width * input_height;
  if (input_size > instance->sift_memory->max_image_size)
  {
    logError(LOG_TAG, "Provided input image size (%d*%d=%d) is greater than the configured maximum image size (%d).", input_width, input_height,
             input_size, instance->sift_memory->max_image_size);
    return false;
  }
  else if (input_size < 1024u)
  {
    logError(LOG_TAG, "Invalid input image size (%d*%d=%d). Input image size must be greater than or equal to 1024", input_width, input_height, input_size,
             instance->sift_memory->max_image_size);
    return false;
  }
  else
  {
    return true;
  }
}

static bool isInputFeatureCoundValid(vksift_Instance instance, const uint32_t nb_feats)
{
  if (nb_feats > instance->sift_memory->max_nb_sift_per_buffer)
  {
    logError(LOG_TAG, "Provided features count (%d) is greater than the configured maximum number of features per GPU buffer size (%d).", nb_feats,
             instance->sift_memory->max_nb_sift_per_buffer);
    return false;
  }
  else
  {
    return true;
  }
}

static bool isInputOctaveIdxValid(vksift_Instance instance, const uint32_t octave_idx)
{
  if (octave_idx >= instance->sift_memory->curr_nb_octaves)
  {
    logError(LOG_TAG, "Requested octave idx is %d but the current number of octaves is %d", octave_idx, instance->sift_memory->curr_nb_octaves);
    return false;
  }
  else
  {
    return true;
  }
}

static bool isInputScaleIdxValid(vksift_Instance instance, const uint32_t scale_idx, bool is_dog)
{
  uint32_t add_scale = is_dog ? 2 : 3;
  if (scale_idx >= (instance->sift_memory->nb_scales_per_octave + add_scale))
  {
    logError(LOG_TAG, "Requested scale idx is %d but the number of %s scales is %d", scale_idx, is_dog ? "DoG" : "blurred",
             (instance->sift_memory->nb_scales_per_octave + add_scale));
    return false;
  }
  else
  {
    return true;
  }
}
// =============================================================================
// IMAS (ASIFT) tilt-simulation API — lazy-creates the pipeline on first call.
// =============================================================================

// Caller must have uploaded the source image to input_image already (via
// vksift_detectFeatures, which doubles as the upload path). Runs the 5-shader
// IMAS chain for (t_factor, theta_rad) and writes the tilted Float32 result
// to the readback buffer; returns dimensions via out_w / out_h.
bool vksift_runImas(vksift_Instance instance, uint32_t input_w, uint32_t input_h,
                    float t_factor, float theta_rad, uint32_t *out_w, uint32_t *out_h)
{
  if (instance == NULL) return false;
  if (instance->imas_pipeline == NULL)
  {
    instance->imas_pipeline = vksift_createImasPipeline(
        instance->vulkan_device, instance->sift_memory, instance->sift_detector->image_sampler);
    if (instance->imas_pipeline == NULL) return false;
  }
  return vksift_runImasWarp(instance->imas_pipeline, input_w, input_h, t_factor, theta_rad, out_w, out_h);
}

// Returns a host-mapped pointer to the Float32 tilted result from the most
// recent vksift_runImas call. Layout: row-major, stride = out_w pixels,
// total = out_w × out_h floats.
const float *vksift_getImasReadbackPtr(vksift_Instance instance)
{
  if (instance == NULL || instance->imas_pipeline == NULL) return NULL;
  return (const float *)instance->imas_pipeline->readback_ptr;
}

// Run SIFT detection on the tilted image produced by the IMAS pipeline,
// reading mem->rotated_image device-side (no host roundtrip).
//
// `canvas_w` × `canvas_h` are the SIFT pyramid input dims — keep these
// stable across warps to avoid per-warp pyramid reallocation (use the
// max tilted canvas at init time).
//
// `valid_w` × `valid_h` are the actual sub-region the IMAS pipeline wrote
// into mem->rotated_image (= out_w, out_h from vksift_runImas). The
// quantize shader writes those pixels from rotated_image and fills the
// rest of input_image with `fill_value` — mirrors the host-roundtrip
// path's pad_tilted layout.
void vksift_detectFeaturesOnImas(vksift_Instance instance,
                                 const uint32_t canvas_w, const uint32_t canvas_h,
                                 const uint32_t valid_w, const uint32_t valid_h,
                                 const float fill_value,
                                 const uint32_t gpu_buffer_id)
{
  if (instance == NULL ||
      !isBufferIdxValid(instance, gpu_buffer_id) ||
      !isInputResolutionValid(instance, canvas_w, canvas_h))
  {
    logError(LOG_TAG, "vksift_detectFeaturesOnImas() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  VkFence fences[2] = {instance->sift_detector->end_of_detection_fence, instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);

  bool memory_layout_updated = false;
  if (!vksift_prepareSiftMemoryForDetection(instance->sift_memory, NULL, canvas_w, canvas_h, gpu_buffer_id, &memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_detectFeaturesOnImas() error: failed to prepare SIFT memory");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
    return;
  }

  // Configure the quantize dispatch's per-warp params. Force a re-record so
  // the push consts in the from-IMAS command buffer pick up the new values
  // (when memory layout didn't change otherwise).
  instance->sift_detector->quantize_valid_w    = valid_w;
  instance->sift_detector->quantize_valid_h    = valid_h;
  instance->sift_detector->quantize_fill_value = fill_value;
  if (!memory_layout_updated)
  {
    instance->sift_detector->pending_warp_dirty = true;
  }

  if (!vksift_dispatchSiftDetectionFromImas(instance->sift_detector, gpu_buffer_id, memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_detectFeaturesOnImas() error: failed to dispatch detection");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

// =============================================================================
// Phase B-3 fused IMAS + Quantize + SIFT-detect entrypoint. Single GPU
// submission per warp; no host roundtrip between IMAS chain and SIFT detect.
// =============================================================================
void vksift_detectFeaturesFusedImas(vksift_Instance instance,
                                    uint32_t W, uint32_t H,
                                    float t_factor, float theta_rad,
                                    uint32_t canvas_w, uint32_t canvas_h,
                                    uint32_t gpu_buffer_id)
{
  if (instance == NULL ||
      !isBufferIdxValid(instance, gpu_buffer_id) ||
      !isInputResolutionValid(instance, canvas_w, canvas_h))
  {
    logError(LOG_TAG, "vksift_detectFeaturesFusedImas() error: invalid input.");
    instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  VkFence fences[2] = {instance->sift_detector->end_of_detection_fence,
                      instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(instance->vulkan_device->device, 2, fences, VK_TRUE, UINT64_MAX);

  // Lazy-create the IMAS pipeline on first fused call. The pre-recorded fused
  // command buffer needs the IMAS pipeline's pipeline + descriptor set objects
  // bound during recording — so once the pipeline is up we must trigger a
  // re-record by marking pending_warp_dirty (which forces recordCommandBuffers
  // in vksift_dispatchFusedImasWarpForSlot).
  bool imas_just_created = false;
  if (instance->imas_pipeline == NULL)
  {
    instance->imas_pipeline = vksift_createImasPipeline(
        instance->vulkan_device, instance->sift_memory, instance->sift_detector->image_sampler);
    if (instance->imas_pipeline == NULL)
    {
      logError(LOG_TAG, "vksift_detectFeaturesFusedImas() error: failed to create IMAS pipeline");
      instance->error_cb_func(VKSIFT_VULKAN_ERROR);
      return;
    }
    // Wire the IMAS pipeline reference into the detector so
    // recFusedImasDetectCmdsForSlot can bind its pipelines + descriptor sets.
    instance->sift_detector->imas_pipeline_ref =
        (struct vksift_ImasPipeline_T *)instance->imas_pipeline;
    imas_just_created = true;
  }

  bool memory_layout_updated = false;
  if (!vksift_prepareSiftMemoryForDetection(instance->sift_memory, NULL, canvas_w, canvas_h,
                                            gpu_buffer_id, &memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_detectFeaturesFusedImas() error: failed to prepare SIFT memory");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
    return;
  }

  // If the pyramid was resized (per-slot images destroyed + recreated), the
  // IMAS pipeline's set-0 descriptor sets reference dead image views. Rewire
  // them before the fused cmd buffer dispatches anything through the IMAS
  // pipelines.
  if (memory_layout_updated || imas_just_created)
  {
    vksift_imasRefreshDescriptorSets(instance->imas_pipeline);
  }

  // First call after lazy IMAS creation: force a re-record so the fused cmd
  // buffer picks up the real IMAS pipeline (the very first recordCommandBuffers
  // call during createSiftDetector produced an empty no-op recording because
  // imas_pipeline_ref was NULL then).
  bool force_record = imas_just_created || memory_layout_updated;

  if (!vksift_dispatchFusedImasWarpForSlot(instance->sift_detector, 0u, gpu_buffer_id,
                                           W, H, t_factor, theta_rad, canvas_w, canvas_h,
                                           force_record))
  {
    logError(LOG_TAG, "vksift_detectFeaturesFusedImas() error: failed to dispatch fused chain");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
  }
}

// =============================================================================
// Phase C-3: parallel-pyramid IMAS+detect dispatch. Submits warps in waves of
// up to nb_pyramid_slots fused command buffers per vkQueueSubmit. Each slot
// writes its features to sift_buffer_arr[slot] (guaranteed independent by
// C-2's sift_buffer_count bump). Host waits one fence per wave, then reads
// per-wave feature counts.
//
// CAVEAT: after a wave finishes, slot s's SIFT buffer holds features for that
// wave's warp s; the NEXT wave overwrites it. Callers that need to keep
// features from multiple waves must download per-wave outside of this entry
// point (the JL FFI does exactly that by chunking the call into one wave per
// invocation). When n_warps ≤ nb_pyramid_slots this is a single wave and the
// caller can safely call vksift_downloadFeatures(instance, ..., slot_idx)
// after this returns.
// =============================================================================
void vksift_dispatchParallelIMAS(vksift_Instance instance,
                                 uint32_t W, uint32_t H,
                                 const vksift_WarpSpec *warps, uint32_t n_warps,
                                 uint32_t canvas_w, uint32_t canvas_h,
                                 uint32_t *out_features_per_warp)
{
  if (instance == NULL || warps == NULL || n_warps == 0u ||
      out_features_per_warp == NULL ||
      !isInputResolutionValid(instance, canvas_w, canvas_h))
  {
    logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: invalid input.");
    if (instance != NULL) instance->error_cb_func(VKSIFT_INVALID_INPUT_ERROR);
    return;
  }

  vksift_SiftDetector detector = instance->sift_detector;
  vksift_SiftMemory   memory   = instance->sift_memory;
  VkDevice            device   = instance->vulkan_device->device;
  const uint32_t      n_slots  = memory->nb_pyramid_slots;

  // Wait on the prior detect/match fences so any previous dispatch (single-
  // warp or wave) has fully retired before we reuse end_of_detection_fence.
  VkFence prior_fences[2] = {detector->end_of_detection_fence,
                             instance->sift_matcher->end_of_matching_fence};
  vkWaitForFences(device, 2, prior_fences, VK_TRUE, UINT64_MAX);

  // Lazy-create the IMAS pipeline on first call (same pattern as
  // vksift_detectFeaturesFusedImas). The fused cmd buffer recording needs
  // imas_pipeline_ref wired up before recordCommandBuffers runs.
  bool imas_just_created = false;
  if (instance->imas_pipeline == NULL)
  {
    instance->imas_pipeline = vksift_createImasPipeline(
        instance->vulkan_device, instance->sift_memory, detector->image_sampler);
    if (instance->imas_pipeline == NULL)
    {
      logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: failed to create IMAS pipeline");
      instance->error_cb_func(VKSIFT_VULKAN_ERROR);
      return;
    }
    detector->imas_pipeline_ref =
        (struct vksift_ImasPipeline_T *)instance->imas_pipeline;
    imas_just_created = true;
  }

  // Prepare the SIFT memory for the stable canvas once — every warp in the
  // schedule shares this canvas (parallel-pyramid plan §9.2). Slot 0's
  // gpu_buffer_id is passed; the canvas / pyramid is shared across slots.
  bool memory_layout_updated = false;
  if (!vksift_prepareSiftMemoryForDetection(memory, NULL, canvas_w, canvas_h,
                                            0u, &memory_layout_updated))
  {
    logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: failed to prepare SIFT memory");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
    return;
  }

  // prepareSiftMemoryForDetection only refreshes buffer-section info for
  // target_buffer_idx=0. The parallel path also writes into sift_buffer_arr[s]
  // for s in [1..n_slots), so make sure each slot's buffer info matches the
  // current canvas resolution before writeDescriptorSets binds the per-octave
  // offsets. Triggers on first parallel dispatch (when slots 1+ still have
  // their init-time square-canvas offsets) and after a memory-layout change.
  for (uint32_t s = 1u; s < n_slots; ++s)
  {
    if (vksift_Memory_refreshBufferInfo(memory, s))
    {
      memory_layout_updated = true;
    }
  }

  // If the pyramid was resized or the IMAS pipeline just came up, rewire all
  // slots' IMAS-pipeline descriptor sets. Without this the IMAS shader bindings
  // would reference dead image views from before the resize.
  if (memory_layout_updated || imas_just_created)
  {
    vksift_imasRefreshDescriptorSets(instance->imas_pipeline);
  }

  // TODO(phase-c-async): async-transfer's slot-0-aliased ownership cmd buffers
  // are wrong for parallel waves (acquire/release_buffer_ownership_command_buffer
  // are recorded against slot 0 only — see recordCommandBuffers). For now,
  // refuse to use the async path even if the device exposes it. Mirror the
  // non-async branch of dispatchDetectionCmdBuffer below.
  if (detector->dev->async_transfer_available)
  {
    logWarning(LOG_TAG,
        "vksift_dispatchParallelIMAS: async-transfer available but unsupported on "
        "the parallel path; falling back to single-queue submission for this call.");
  }

  // Force a re-record on the very first wave after IMAS-pipeline lazy init or
  // a memory-layout change. Subsequent waves reuse the existing recording.
  bool need_record = imas_just_created || memory_layout_updated;
  if (!vksift_ensureDetectorCmdBuffersRecorded(detector, 0u, need_record))
  {
    logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: failed to (re)record command buffers");
    instance->error_cb_func(VKSIFT_VULKAN_ERROR);
    return;
  }

  // Mark every slot's sift buffer as not-packed; the detection pipeline writes
  // raw counts into sift_count_staging, so the cached nb_stored_feats from a
  // prior pack (e.g. a matcher run) would shadow the new counts otherwise.
  // prepareSiftMemoryForDetection only flips is_packed for target_buffer_idx,
  // which we always pass as 0u above — handle the rest here.
  for (uint32_t s = 1u; s < n_slots; ++s)
  {
    memory->sift_buffers_info[s].is_packed = false;
  }

  // Process warps in waves of min(n_slots, remaining). One vkQueueSubmit per
  // wave (single batched VkSubmitInfo array), one fence per wave. Reuses the
  // detector's end_of_detection_fence between waves — safe because we wait
  // before each reset.
  VkSubmitInfo submits[VKSIFT_MAX_PYRAMID_SLOTS];
  for (uint32_t base = 0u; base < n_warps; base += n_slots)
  {
    uint32_t remaining = n_warps - base;
    uint32_t wave = (remaining < n_slots) ? remaining : n_slots;

    // (a) Fill each slot's per-warp UBO + dispatch buffer.
    for (uint32_t s = 0u; s < wave; ++s)
    {
      vksift_fillFusedWarpState(detector, s, base + s, W, H,
                                warps[base + s].t_factor, warps[base + s].theta_rad,
                                canvas_w, canvas_h);
    }

    // (b) Build wave VkSubmitInfo entries — one per slot's fused cmd buffer.
    for (uint32_t s = 0u; s < wave; ++s)
    {
      submits[s] = (VkSubmitInfo){
          .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
          .pNext = NULL,
          .waitSemaphoreCount = 0,
          .pWaitSemaphores = NULL,
          .pWaitDstStageMask = NULL,
          .commandBufferCount = 1,
          .pCommandBuffers = &detector->fused_imas_detect_command_buffer[s],
          .signalSemaphoreCount = 0,
          .pSignalSemaphores = NULL};
    }
    vkResetFences(device, 1, &detector->end_of_detection_fence);
    if (vkQueueSubmit(detector->general_queue, wave, submits,
                      detector->end_of_detection_fence) != VK_SUCCESS)
    {
      logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: vkQueueSubmit failed for wave starting at warp %u", base);
      instance->error_cb_func(VKSIFT_VULKAN_ERROR);
      return;
    }
    vkWaitForFences(device, 1, &detector->end_of_detection_fence, VK_TRUE, UINT64_MAX);

    // (c) Read per-slot feature counts from sift_count_staging_buffer. These
    // remain valid until the next wave on the same slot dispatches.
    for (uint32_t s = 0u; s < wave; ++s)
    {
      uint32_t feat_count = 0u;
      if (!vksift_Memory_getBufferFeatureCount(memory, s, &feat_count))
      {
        logError(LOG_TAG, "vksift_dispatchParallelIMAS() error: failed to read feature count for slot %u (warp %u)", s, base + s);
        instance->error_cb_func(VKSIFT_VULKAN_ERROR);
        return;
      }
      out_features_per_warp[base + s] = feat_count;
    }
  }
}
