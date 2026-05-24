#ifndef VULKAN_SIFT_H
#define VULKAN_SIFT_H

#ifdef __cplusplus
extern "C"
{
#endif

#if defined(_WIN32) || defined(_WIN64)
#define VKSIFT_EXPORT __declspec(dllexport)
#else
#define VKSIFT_EXPORT __attribute__((__visibility__("default")))
#endif

#include "vulkansift/vulkansift_types.h"

#include <stdbool.h>
#include <stdint.h>

  // Load/Unload the Vulkan API.
  // vksift_loadVulkan() must be called before the first VulkanSift function call and vksift_unloadVulkan() must be called after that last
  // VulkanSift function call.
  VKSIFT_EXPORT vksift_Result vksift_loadVulkan();
  VKSIFT_EXPORT void vksift_unloadVulkan();

  // Retrieve the name of the available GPU(s) (GPUs must support Vulkan to be visible)
  // If gpu_names is NULL, fill gpu_count with the number of available GPU.
  // If gpu_names is not NULL, copy a number of VKSIFT_GPU_NAMES defined by gpu_count to the gpu_names buffer.
  VKSIFT_EXPORT void vksift_getAvailableGPUs(uint32_t *gpu_count, VKSIFT_GPU_NAME *gpu_names);
  VKSIFT_EXPORT void vksift_setLogLevel(const vksift_LogLevel level);

  // Create and destroy a vksift_Instance. A vksift_Instance manages GPU resources, detection and matching pipelines as configured
  // by the vksift_Config user configuration. It uses only one GPU device specified in the configuration (if not specified
  // the best available GPU should be automatically selected).
  typedef struct vksift_Instance_T *vksift_Instance;
  VKSIFT_EXPORT vksift_Result vksift_createInstance(vksift_Instance *instance_ptr, const vksift_Config *config);
  VKSIFT_EXPORT void vksift_destroyInstance(vksift_Instance *instance_ptr);
  VKSIFT_EXPORT vksift_Config vksift_getDefaultConfig();

  /**
   * Detection/Matching pipelines
   *
   * vksift_detectFeatures() and vksift_matchFeatures() calls DO NOT WAIT for the results to be available.
   * This means that the functions return as soon as the detection/matching pipelines are started without waiting
   * for the results to be available. However, is a detection/matching pipeline is already running, it must wait
   * the end of the previous pipeline before starting the new one.
   **/

  // Set the pending AffineWarp matrix used by the next vksift_detectFeatures() call.
  // The 2x3 matrix is the INVERSE affine that maps warped output pixel coords to input
  // pixel coords (the same convention as the AffineWarp.comp shader):
  //   col_in = a11*col_out + a12*row_out + a13
  //   row_in = a21*col_out + a22*row_out + a23
  // fill_value (normalized [0..1]) is returned for out-of-bounds samples.
  // Identity matrix (a11=a22=1, others 0) reproduces the standard non-ASIFT path.
  VKSIFT_EXPORT void vksift_setPendingAffineWarpInstance(vksift_Instance instance,
                                                        float a11, float a12, float a13,
                                                        float a21, float a22, float a23,
                                                        float fill_value);

  // Set the σ_aa pre-blur applied to the input image before AffineWarp on the
  // next vksift_detectFeatures() call. Direction (dir_x, dir_y) is the 1D blur
  // axis in input pixel coords (ASIFT uses (sin φ, cos φ) for the squash
  // direction). σ = 0 yields a pass-through copy (identity-equivalent).
  VKSIFT_EXPORT void vksift_setPendingPreBlurInstance(vksift_Instance instance,
                                                     float sigma, float dir_x, float dir_y);

  // Copy the image to the GPU and start the detection pipeline on the GPU. Detected features will be stored on the
  // specified GPU buffer. The parameter image_data must point to an array of uint8_t values representing a grayscale image (row-major).
  VKSIFT_EXPORT void vksift_detectFeatures(vksift_Instance instance, const uint8_t *image_data, const uint32_t image_width, const uint32_t image_height,
                                           const uint32_t gpu_buffer_id);

  // Run the IMAS (ASIFT) tilt-simulation chain on the input image currently
  // residing in the instance's input_image buffer (uploaded by the most recent
  // vksift_detectFeatures call). Lazy-creates the IMAS pipeline on first call.
  // Outputs the tilted Float32 result to a host-mapped readback buffer; caller
  // reads (out_w × out_h) floats from vksift_getImasReadbackPtr().
  VKSIFT_EXPORT bool vksift_runImas(vksift_Instance instance,
                                    uint32_t input_w, uint32_t input_h,
                                    float t_factor, float theta_rad,
                                    uint32_t *out_w, uint32_t *out_h);
  VKSIFT_EXPORT const float *vksift_getImasReadbackPtr(vksift_Instance instance);

  // Cross-warp ellipse-footprint NMS over an aggregated pool of features.
  // Caller writes `n_in` vksift_Feature records into the buffer returned by
  // vksift_getCrossWarpInPtr(instance) (the buffer carries a u32 nb / u32 max_nb
  // header followed by the feature array — the helper sets both).  After this
  // returns true, vksift_getCrossWarpOutPtr(instance) is the survivor buffer
  // (same header layout); *n_out holds the survivor count.  canvas_w/canvas_h
  // size the splat scratch image (typically input image dims).
  //
  // n_in is internally clamped to 65535 due to the 16-bit feature-index packing
  // in CrossWarpSplat.comp.  Caller should pre-cap if more features arrive.
  // mode: 0 = M×M centroid+window NMS, 1 = Mahalanobis ellipse-footprint NMS.
  VKSIFT_EXPORT bool vksift_runCrossWarpNmsInstance(vksift_Instance instance,
                                                    uint32_t canvas_w, uint32_t canvas_h,
                                                    uint32_t n_in, uint32_t *n_out,
                                                    int32_t window_half,
                                                    int32_t mode, float k_cutoff);
  VKSIFT_EXPORT void *vksift_getCrossWarpInPtr(vksift_Instance instance);
  VKSIFT_EXPORT void *vksift_getCrossWarpOutPtr(vksift_Instance instance);
  VKSIFT_EXPORT uint32_t vksift_getCrossWarpCapacity(vksift_Instance instance);

  // Run SIFT detection on the most-recent IMAS-tilted image (mem->rotated_image,
  // device-side) — no host roundtrip. Caller must have run vksift_runImas first
  // to populate rotated_image.
  //
  // (canvas_w, canvas_h) = SIFT pyramid input dims — keep stable across warps
  //   to avoid per-warp pyramid reallocation. Pass the same (W, H) used at
  //   init for the entire IMAS-25 schedule.
  // (valid_w, valid_h)   = the sub-region the IMAS pipeline wrote into
  //   rotated_image (= out_w, out_h returned by vksift_runImas). The quantize
  //   shader writes those pixels from rotated_image and fills the rest of
  //   input_image with `fill_value`.
  VKSIFT_EXPORT void vksift_detectFeaturesOnImas(vksift_Instance instance,
                                                 const uint32_t canvas_w, const uint32_t canvas_h,
                                                 const uint32_t valid_w, const uint32_t valid_h,
                                                 const float fill_value,
                                                 const uint32_t gpu_buffer_id);

  // Phase B-3: fully-fused IMAS + Quantize + SIFT-detect in a single GPU
  // submission. Lazily creates the IMAS pipeline on first call. The fused
  // command buffer is pre-recorded per slot; this call updates the per-slot
  // WarpParamsUBO and indirect-dispatch buffer, then submits the recording.
  // No host roundtrip between IMAS warp and SIFT detect.
  //
  // (W, H) are the input-image dims (must match the dims used to upload via
  // vksift_detectFeatures). (t_factor, theta_rad) define the IMAS warp.
  // (canvas_w, canvas_h) is the stable SIFT pyramid input canvas — keep
  // identical across warps in a schedule to avoid pyramid reallocation.
  // gpu_buffer_id selects the destination SIFT buffer.
  //
  // Note: Phase B-3 only exercises pyramid slot 0; the SIFT-detect descriptor
  // sets are still bound to mem->slots[0]'s image views. Phase C will allow
  // parallel waves across slots.
  VKSIFT_EXPORT void vksift_detectFeaturesFusedImas(vksift_Instance instance,
                                                    uint32_t W, uint32_t H,
                                                    float t_factor, float theta_rad,
                                                    uint32_t canvas_w, uint32_t canvas_h,
                                                    uint32_t gpu_buffer_id);

  // Phase C-3: parallel-pyramid IMAS+detect dispatch. Lazily creates the IMAS
  // pipeline on first call (same as vksift_detectFeaturesFusedImas) and
  // dispatches the supplied warp schedule across nb_pyramid_slots in waves.
  //
  // Per wave: each slot s in [0..wave) gets its WarpParamsUBO + dispatch buffer
  // filled for warps[base+s], then all wave fused command buffers are submitted
  // in a single vkQueueSubmit (batch). The host waits one fence per wave;
  // after the wait, slot s's feature count is copied into
  // out_features_per_warp[base+s] and slot s's sift_buffer_arr[s] holds the
  // features for that warp. Callers should download features immediately after
  // each wave because the next wave overwrites the slot's SIFT buffer.
  //
  // (canvas_w, canvas_h) must be stable across all warps in a single call.
  // out_features_per_warp must be sized at least n_warps.
  typedef struct {
    float t_factor;
    float theta_rad;
    // Global warp index in the caller's IMAS schedule. Stamped into the
    // WarpParamsUBO so BackProjectFeatures.comp can recognise the identity
    // warp (warp_idx == 0) and skip the K·σ·σ_max boundary check. The
    // dispatcher otherwise can't recover this — it loops wave by wave with
    // a local base offset, so without an explicit field every wave's UBO
    // would receive warp_idx == 0.
    uint32_t warp_idx;
  } vksift_WarpSpec;

  VKSIFT_EXPORT void vksift_dispatchParallelIMAS(
      vksift_Instance instance,
      uint32_t W, uint32_t H,
      const vksift_WarpSpec *warps, uint32_t n_warps,
      uint32_t canvas_w, uint32_t canvas_h,
      uint32_t *out_features_per_warp);

  // For each SIFT feature in the buffer A, find the 2-nearest neighbors in the buffer B, store feature index and descriptors L2 distance
  // for the two neighbors.
  VKSIFT_EXPORT void vksift_matchFeatures(vksift_Instance instance, const uint32_t gpu_buffer_id_A, const uint32_t gpu_buffer_id_B);

  /**
   * Data transfer functions
   *
   * All transfer functions are BLOCKING. This means that the functions return when the requested data is made available to the user.
   * If the requested data is currently being used or generated by the detection/matching pipeline the function will block until
   * the detection/matching process is done.
   *
   * Note: if your GPU support async-tranfer operations, the data-transfer can be executed by the GPU while another pipeline is running,
   * otherwise the GPU may complete other operations before doing the transfer (defined by vendor).
   **/

  // Return the number of features available in the specified GPU buffer.
  VKSIFT_EXPORT uint32_t vksift_getFeaturesNumber(vksift_Instance instance, const uint32_t gpu_buffer_id);

  // Download SIFT features from the specified GPU buffer and copy them to feats_ptr. The parameter feats_ptr must provide enough space to
  // store all the features present in the SIFT buffer (number of features retrieved with vksift_getFeaturesNumber())
  VKSIFT_EXPORT void vksift_downloadFeatures(vksift_Instance instance, vksift_Feature *feats_ptr, const uint32_t gpu_buffer_id);

  // Upload SIFT features to the specified GPU buffer.
  VKSIFT_EXPORT void vksift_uploadFeatures(vksift_Instance instance, const vksift_Feature *feats_ptr, const uint32_t nb_feats,
                                           const uint32_t gpu_buffer_id);

  // Return the number of matches found.
  VKSIFT_EXPORT uint32_t vksift_getMatchesNumber(vksift_Instance instance);

  // Copy GPU matches information (vksift_Match_2NN) to the matches buffer. The parameter matches must provide enough space to store
  // all the matches (number of matches retrieved with vksift_getMatchesNumber())
  VKSIFT_EXPORT void vksift_downloadMatches(vksift_Instance instance, vksift_Match_2NN *matches);

  // Get the buffer availability status. Return true if the GPU is not using the buffer for a detection/matching task, false otherwise.
  // Can be used to check for the result or device availability and avoid long CPU blocking calls.
  VKSIFT_EXPORT bool vksift_isBufferAvailable(vksift_Instance instance, const uint32_t gpu_buffer_id);

  // Scale-space access functions (for debug and visualization)
  // Return the current number of octave used (depends on configuration and input image resolution).
  VKSIFT_EXPORT uint8_t vksift_getScaleSpaceNbOctaves(vksift_Instance instance);
  // Return the image resolution used for the specified octave.
  VKSIFT_EXPORT void vksift_getScaleSpaceOctaveResolution(vksift_Instance instance, const uint8_t octave, uint32_t *octave_images_width,
                                                          uint32_t *octave_images_height);
  // Copy the selected Gaussian image data to the blurred_image float buffer. (scale value in [0, config.nb_scales_per_octave+3[ )
  VKSIFT_EXPORT void vksift_downloadScaleSpaceImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *blurred_image);
  // Copy the selected Difference of Gaussian image data to the blurred_image float buffer. (scale value in [0, config.nb_scales_per_octave+2[ )
  VKSIFT_EXPORT void vksift_downloadDoGImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *dog_image);

  /**
   * GPU Debug functions
   *
   * WARNING | Only available when use_gpu_debug_functions is true and gpu_debug_external_window_info is filled
   *         | in the configuration structure during the call to vksift_createInstance.
   *         | Do nothing and print a warning if this is not the case.
   **/
  // Draw an empty frame in the debug window. Required to use graphics GPU debuggers/profilers such as RenderDoc or Nvidia Nsight
  // (They use frame delimiters to detect when to start/stop debugging and can't detect compute-only applications)
  VKSIFT_EXPORT void vksift_presentDebugFrame(vksift_Instance instance);

#ifdef __cplusplus
}
#endif

#endif // VULKAN_SIFT_H