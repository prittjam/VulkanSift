#ifndef VKSIFT_TYPES_H
#define VKSIFT_TYPES_H

#ifdef __cplusplus
extern "C"
{
#endif

#define VKSIFT_FEATURE_NB_HIST 4
#define VKSIFT_FEATURE_NB_ORI 8

#include <stdbool.h>
#include <stdint.h>

  typedef char VKSIFT_GPU_NAME[256];

  typedef struct
  {
    float x; // (x,y) if the keypoint position in the input image
    float y;

    float scale_x; // (scale_x,scale_y) is the keypoint position in the pyramid image where it was detected
    float scale_y;
    uint32_t scale_idx; // index of the gaussian scale image
    int32_t octave_idx; // index of the octave (-1 correspond to the upscaled image)
    float sigma;        // blur level of the gaussian scale image (this value is divided by two if upsampling was used)
    float orientation;  // feature orientation (rad)
    float intensity;    // keypoint pixel intensity in the Difference of Gaussian image

    uint8_t descriptor[VKSIFT_FEATURE_NB_HIST * VKSIFT_FEATURE_NB_HIST * VKSIFT_FEATURE_NB_ORI]; // Feature descriptor
  } vksift_Feature;

  typedef struct
  {
    uint32_t idx_a;  // feature index in the feature set A
    uint32_t idx_b1; // nearest neighbor feature index in the feature set B
    uint32_t idx_b2; // second nearest neighbor feature index in the feature set B
    float dist_a_b1; // descriptors L2 distance between the set A feature and the nearest neighbor
    float dist_a_b2; // same as dist_a_b1 but with the second nearest neighbor
  } vksift_Match_2NN;

  typedef enum
  {
    VKSIFT_NO_LOG,
    VKSIFT_LOG_ERROR,
    VKSIFT_LOG_WARNING,
    VKSIFT_LOG_INFO,
    VKSIFT_LOG_DEBUG
  } vksift_LogLevel;

  typedef enum
  {
    VKSIFT_DESCRIPTOR_FORMAT_UBC,
    VKSIFT_DESCRIPTOR_FORMAT_VLFEAT
  } vksift_DescriptorFormat;

  typedef enum
  {
    VKSIFT_PYRAMID_PRECISION_FLOAT32,
    VKSIFT_PYRAMID_PRECISION_FLOAT16
  } vksift_PyramidPrecisionMode;

  typedef enum
  {
    VKSIFT_SUCCESS,

    // VKSIFT_INVALID_INPUT_ERROR errors are detected and returned early in the functions (before anything else was done).
    // The state of the vksift_Instance is not affected and the instance can still be used.
    VKSIFT_INVALID_INPUT_ERROR,

    // VKSIFT_VULKAN_ERROR are errors related to the GPU and Vulkan API (like out of memory errors).
    // After receiving this type of error, the vksift_Instance is invalid and should be directly destroyed.
    VKSIFT_VULKAN_ERROR
  } vksift_Result;

  ///////////////////////////////
  // For GPU debugging only
  // To be able to render frames and make the SIFT pipelines visible for debugger/profiler, the user need to create a window (or use the available
  // window on Android) and provide the information to vksift. Since this is different depending on which window provider you use, informations must be
  // filled in the structure vksift_ExternalWindowInfo with opaque type members. Depending on your window manager, fill the two variables with the
  // variables from your Window with the types described below:
  // Xlib (Linux):
  //    - context = Display**
  //    - window = Window*
  // Win32 (Windows):
  //    - context = HINSTANCE*
  //    - window = HWND*
  // Android:
  //    - context = NULL
  //    - window = ANativeWindow**
  typedef struct
  {
    void *context;
    void *window;
  } vksift_ExternalWindowInfo;

  typedef struct
  {
    // Input/Output configuration

    // Maximum size (in pixels) for the input grayscale images.
    //
    // Two ways to express this:
    //   - Legacy: set `input_image_max_size = max_width * max_height` and leave
    //     `input_image_max_width / _height` at 0. SIFT allocates a SQUARE buffer
    //     of side ceil(sqrt(input_image_max_size)) — fine for square inputs but
    //     over-budgets when the max output is non-square (e.g. an ASIFT tilted
    //     image at W_rot × H_rot/t with W_rot ≫ H_rot/t).
    //   - Explicit: set `input_image_max_width` and `input_image_max_height` to
    //     the actual maximum dims you'll ever pass. `input_image_max_size` is
    //     re-derived as the product. No square over-budgeting.
    //
    // Default: 1920×1080 = legacy form for backward compat.
    uint32_t input_image_max_size;
    // Explicit non-square sizing. Both must be > 0 to take effect; otherwise
    // the legacy square ceiling is used.
    uint32_t input_image_max_width;
    uint32_t input_image_max_height;
    // Number of SIFT buffers (stored on the GPU) to be reserved by the application (default: 2)
    uint32_t sift_buffer_count;
    // Maximum number of SIFT features stored by a GPU SIFT bufer (default: 100000)
    uint32_t max_nb_sift_per_buffer;

    // Number of independent pyramid slots used by the parallel IMAS+detect
    // pipeline (default: 1). Each slot owns its own per-pyramid GPU memory
    // (rotated_image, input_image, blurred/warped, octave arrays, DoG arrays,
    // blur tmp, sift_buffer) — allowing N IMAS+detect chains to be submitted
    // simultaneously and overlap on the GPU scheduler. Memory cost: ~3-4 GB
    // per slot at a 4K input with upsample=true; 5 slots fit in a 24 GB
    // 4090 with headroom. Set to 1 to disable parallel processing (single
    // pyramid, classic behavior).
    uint32_t nb_pyramid_slots;

    // SIFT algorithm configuration

    // If true, a 2x upscaled version of the input image will used for the Gaussian scale-space construction.
    // Using this option, more SIFT features will be found at the expense of a longer processing time. (default: true)
    bool use_input_upsampling;
    // Number of octaves used in the Gaussian scale-space. If set to 0, the number of octaves is defined by the implementation and
    // depends on the input image resolution. (default: 0)
    uint8_t nb_octaves;
    // Number of scales used per octave in the Gaussian scale-space. (default: 3)
    uint8_t nb_scales_per_octave;
    // Assumed blur level for the input image (default: 0.5f)
    float input_image_blur_level;
    // Blur level for the Gaussian scale-space seed scale (default: 1.6f)
    float seed_scale_sigma;
    // Minimum Difference of Gaussian image intensity threshold used to detect a SIFT keypoints (expressed in normalized intensity value [0.f..1.f])
    // In the implementation, this value is divided by nb_scales_per_octave before being used. (default: 0.04f)
    float intensity_threshold;
    // Edge threshold used to discard SIFT keypoints on Difference of Gaussian image edges (default: 10.f)
    float edge_threshold;
    // Max number of orientation per SIFT keypoint (one descriptor is detector for each orientation
    // If set to 0, no limit will be applied. (default: 0)
    uint32_t max_nb_orientation_per_keypoint;
    // Output format for the descriptor (as an example with other implementations, VLFeat format is used by VLFeat and PopSift and UBC wormat is used by
    // Lowe's implementation, OpenCV and SiftGPU). (For the format comparison, check https://www.vlfeat.org/overview/sift.html)
    // (default: VKSIFT_DESCRIPTOR_FORMAT_UBC)
    vksift_DescriptorFormat descriptor_format;

    // If true, skip orientation assignment and descriptor computation. Only detect keypoints.
    // Useful when only keypoint locations and scores are needed. (default: false)
    bool detection_only;
    // If true, use 2D spatial NMS (8 neighbors per scale) instead of 3D NMS (26 neighbors across 3 scales).
    // No subpixel refinement is performed in 2D mode. (default: false)
    bool use_2d_nms;
    // If true, detectFeatures expects RGBA data (4 bytes/pixel) instead of grayscale (1 byte/pixel).
    // The RGBA→Gray conversion is performed on the GPU via a compute shader. (default: false)
    bool use_rgba_input;
    // If true, detectFeatures expects RGB data (3 bytes/pixel) instead of grayscale (1 byte/pixel).
    // The RGB→Gray conversion is performed on the GPU via a compute shader. (default: false)
    bool use_rgb_input;

    // GPU and implementation configuration

    // Define the GPU used by the Instance. For a given GPU, the device index should the same as its corresponding name index when retrieved
    // with vksift_getAvailableGPUs(). If gpu_device_index<0 the GPU with highest expected performance is selected. (default: -1)
    int32_t gpu_device_index;
    // If true, the GPU hardware texture samplers will be used to speed up the Gaussian scale-space construction. (default: true)
    bool use_hardware_interpolated_blur;
    // Defines the scale-space image format precision (default: VKSIFT_PYRAMID_PRECISION_FLOAT32), images being the heaviest GPU resource
    // switching to a VKSIFT_PYRAMID_PRECISION_FLOAT16 reduces the GPU memory usage by a factor of two, with a small impact on the feature quality.
    vksift_PyramidPrecisionMode pyramid_precision_mode;

    // Error function that can be called by all VulkanSift functions (except vksift_destroyX functions and functions returning a vksift_Result).
    // This function is called when errors (invalid input arguments, Vulkan function failures) are detected by the VulkanSift function.
    // Can be used by C++ users to throw exceptions inside the callback. See the vksift_Result description for information on what can be used/done
    // after receiving errors. (default: wrapper around abort() function)
    void (*on_error_callback_function)(vksift_Result);

    // GPU configuration to use GPU debugger/profilers

    // Set to true and fill the data in external_window_info to be able to use vksift_presentDebugFrame() and profile applications
    // with GPU debuggers/profilers. (default: false)
    bool use_gpu_debug_functions;
    // Check the vksift_ExternalWindowInfo for information on how to fill this structure. (default: {context=NULL, window=NULL})
    vksift_ExternalWindowInfo gpu_debug_external_window_info;

  } vksift_Config;

#ifdef __cplusplus
}
#endif

#endif // VKSIFT_TYPES_H
