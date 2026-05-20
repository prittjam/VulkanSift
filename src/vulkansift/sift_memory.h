#ifndef VKSIFT_SIFTMEMORY
#define VKSIFT_SIFTMEMORY

#include "vkenv/vulkan_device.h"
#include "vulkansift/vulkansift_types.h"

// Maximum number of independent pyramid slots the parallel IMAS+detect path
// can be configured for. Runtime nb_pyramid_slots ≤ this. Used to size the
// per-slot static array on vksift_SiftMemory_T.
#define VKSIFT_MAX_PYRAMID_SLOTS 8u

typedef struct
{
  // Buffer memory can be arranged in two different ways. After being filled with a detection pipeline the buffer is arranged in sections
  // that contain the SIFT features detected for each octave. Each section has a header containing the number of features found and the max
  // number of features that can be stored in this section.
  // If the buffer is used for the matching pipeline or is filled from user data, it will be created or moved to a "packed" (to the left) format
  // where there are no sections. A single header containing the number and max number of features is followed by all the SIFT features.
  bool is_packed;
  uint32_t nb_stored_feats; // only available if is_packed is true

  uint32_t curr_input_width;
  uint32_t curr_input_height;
  // Number of SIFT feature per sections
  uint32_t *octave_section_max_nb_feat_arr;
  // Offset to get the start of the sections
  VkDeviceSize *octave_section_offset_arr;
  // Section sizes
  VkDeviceSize *octave_section_size_arr;
} vksift_SiftBufferInfo;

typedef struct
{
  uint32_t width;
  uint32_t height;
} vksift_OctaveResolution;

// Per-pyramid GPU resources. The parallel IMAS+detect pipeline keeps
// nb_pyramid_slots of these so N pyramids can be in-flight simultaneously.
// Phase A only consumes slots[0]; Phase B+ adds parallel scheduling.
typedef struct
{
  // Pyramid input (R8_UNORM). Either uploaded host-side via image_staging_buffer
  // (regular detect path) or written device-side by the QuantizeF32ToInput
  // shader (on-IMAS detect path).
  VkImage input_image;
  VkImageView input_image_view;
  VkDeviceMemory input_image_memory;
  VkDeviceSize input_image_memory_size;

  // Pre-blurred input image (r32f). Output of the per-warp PreBlur1D.comp
  // pass that applies the Morel-Yu σ_aa = 0.8·√(t²−1) anti-alias filter
  // along the warp's squash direction. Sized at the input resolution; serves
  // as the sampler source for the AffineWarp pass below. At σ ≈ 0 the
  // PreBlur1D shader degenerates to a pass-through copy of input_image.
  VkImage blurred_input_image;
  VkImageView blurred_input_image_view;
  VkDeviceMemory blurred_input_image_memory;
  VkDeviceSize blurred_input_image_memory_size;

  // Warped input image (r32f). Output of the optional AffineWarp.comp pass;
  // becomes the source of the first Gaussian blur for ASIFT-style detection.
  // Sized at curr_input_image_width × curr_input_image_height to fit the
  // worst-case tilt-warp output bbox.
  VkImage warped_input_image;
  VkImageView warped_input_image_view;
  VkDeviceMemory warped_input_image_memory;
  VkDeviceSize warped_input_image_memory_size;

  // Rotated working image (r32f) sized at WORST-CASE rotated dimensions for
  // the IMAS-25 covering: max(W_rot × H_rot) ≈ (W + H) × (W + H). Used as a
  // scratch buffer holding the chain output of:
  //   AffineWarp(rotation only)  → rotated image
  //   GaussianBlur(σ_aa vertical)→ blurred-rotated image (in-place)
  // Matches `frot` + `GaussianBlur1D` from libSimuTilts/digital_tilt.cpp.
  VkImage rotated_image;
  VkImageView rotated_image_view;
  VkDeviceMemory rotated_image_memory;
  VkDeviceSize rotated_image_memory_size;
  uint32_t rotated_image_max_width;
  uint32_t rotated_image_max_height;

  // Final tilted image (r32f) sized at W_rot × ⌊H_rot/t⌋ for the IMAS-25
  // covering's smallest non-identity tilt. Output of FprojBilinearY.comp;
  // serves as the detection-pipeline input for per-warp pyramid construction.
  VkImage tilted_image;
  VkImageView tilted_image_view;
  VkDeviceMemory tilted_image_memory;
  VkDeviceSize tilted_image_memory_size;
  uint32_t tilted_image_max_width;
  uint32_t tilted_image_max_height;

  // RGBA input image (only allocated when use_rgba_input=true)
  VkImage rgba_input_image;
  VkImageView rgba_input_image_view;
  VkDeviceMemory rgba_input_image_memory;
  VkDeviceSize rgba_input_image_memory_size;

  // RGB input buffer (only allocated when use_rgb_input=true)
  // Uses SSBO since VK_FORMAT_R8G8B8_UNORM has poor storage image support
  VkBuffer rgb_input_buffer;
  VkDeviceMemory rgb_input_buffer_memory;
  VkDeviceSize rgb_input_buffer_size;

  // Scale-space pyramid (per-octave arrays). Sized max_nb_octaves at slot
  // allocation time; entries past curr_nb_octaves are unused.
  VkImage *octave_image_arr;
  VkImageView *octave_image_view_arr;
  VkDeviceMemory *octave_image_memory_arr;
  VkDeviceSize *octave_image_memory_size_arr;

  VkImage *blur_tmp_image_arr;
  VkImageView *blur_tmp_image_view_arr;
  VkDeviceMemory *blur_tmp_image_memory_arr;
  VkDeviceSize *blur_tmp_image_memory_size_arr;

  VkImage *octave_DoG_image_arr;
  VkImageView *octave_DoG_image_view_arr;
  VkDeviceMemory *octave_DoG_image_memory_arr;
  VkDeviceSize *octave_DoG_image_memory_size_arr;
} vksift_SiftPyramidSlot;

typedef struct vksift_SiftMemory_T
{
  vkenv_Device device; // parent device

  VkCommandPool general_command_pool;
  VkCommandPool async_transfer_command_pool;
  VkCommandBuffer transfer_command_buffer;

  VkFence transfer_fence;

  // SIFT buffers
  vksift_SiftBufferInfo *sift_buffers_info;
  VkBuffer *sift_buffer_arr;
  VkDeviceMemory *sift_buffer_memory_arr;
  VkBuffer *sift_count_staging_buffer_arr;
  VkDeviceMemory *sift_count_staging_buffer_memory_arr;
  void **sift_count_staging_buffer_ptr_arr;
  VkBuffer sift_staging_buffer;
  VkDeviceMemory sift_staging_buffer_memory;
  void *sift_staging_buffer_ptr;

  // Pyramid objects
  VkBuffer image_staging_buffer;
  VkDeviceMemory image_staging_buffer_memory;
  void *image_staging_buffer_ptr;

  // Per-pyramid resources. Phase A: only slots[0] is consumed by detector +
  // IMAS pipeline. Phase B+: parallel IMAS waves use slots[s] for s in
  // [0, nb_pyramid_slots). Setup + destroy loop over slots[0 .. nb_pyramid_slots).
  vksift_SiftPyramidSlot slots[VKSIFT_MAX_PYRAMID_SLOTS];
  uint32_t nb_pyramid_slots;

  // Cached copy of slots[0].input_image, kept in sync by recCopyInputImageCmds
  // whenever a regular vksift_detectFeatures uploads new content. The IMAS
  // pipeline reads from cached_input_image_view (rather than the per-slot
  // input_image_view) so that the device-side on-IMAS detect path (which
  // overwrites input_image with quantized tilted content) doesn't corrupt the
  // IMAS source between warps. Single shared instance — read-only IMAS source.
  VkImage cached_input_image;
  VkImageView cached_input_image_view;
  VkDeviceMemory cached_input_image_memory;
  VkDeviceSize cached_input_image_memory_size;

  bool use_rgba_input;
  bool use_rgb_input;

  VkImage output_image; // output image is used to export scalespace images to the CPU for debug/viz
  VkDeviceMemory output_image_memory;

  // Pyramid info
  uint32_t curr_input_image_width;
  uint32_t curr_input_image_height;
  uint32_t curr_nb_octaves;
  vksift_OctaveResolution *octave_resolutions;

  // Matches buffers
  uint32_t curr_nb_matches; // simply updated using the buffer A number of features
  VkBuffer match_output_buffer;
  VkDeviceMemory match_output_buffer_memory;

  VkBuffer match_output_staging_buffer;
  VkDeviceMemory match_output_staging_buffer_memory;
  void *match_output_staging_buffer_ptr;

  // Other
  VkDeviceSize *indirect_oridesc_offset_arr;
  VkBuffer indirect_orientation_dispatch_buffer;
  VkDeviceMemory indirect_orientation_dispatch_buffer_memory;
  VkBuffer indirect_descriptor_dispatch_buffer;
  VkDeviceMemory indirect_descriptor_dispatch_buffer_memory;

  VkQueue general_queue;
  VkQueue async_transfer_queue;

  // Config
  uint32_t max_image_size;
  uint32_t max_nb_octaves;
  uint32_t nb_scales_per_octave;
  uint32_t nb_sift_buffer;
  uint32_t max_nb_sift_per_buffer;
  vksift_PyramidPrecisionMode pyr_precision_mode;
  bool use_upsampling;
} * vksift_SiftMemory;

// Setup every memory object
// Create image/buffers with the maximum size requirements for memory allocation
// Map staging in and staging out buffers (input image, output image, input buffer, output buffer)
bool vksift_createSiftMemory(vkenv_Device device, vksift_SiftMemory *memory_ptr, const vksift_Config *config);

// Recompute the octave resolution (and number) and recreate+bind the images
// Since the images will be new GPU objects, the related descriptors in the computing pipelines must be updated
bool vksift_prepareSiftMemoryForDetection(vksift_SiftMemory memory, const uint8_t *image_data, const uint32_t input_width, const uint32_t input_height,
                                          const uint32_t target_buffer_idx, bool *memory_layout_updated);

// Update the buffer structure to a packed format with a 2-uint32 header (containing the number of features) and all the features aligned
// after this header (requirement for the matching pipeline)
bool vksift_prepareSiftMemoryForMatching(vksift_SiftMemory memory, const uint32_t target_buffer_A_idx, const uint32_t target_buffer_B_idx);

// Read from staging buffer memory to retrieve the number of features stored in a SIFT buffer (GPU not involved in this function)
bool vksift_Memory_getBufferFeatureCount(vksift_SiftMemory memory, const uint32_t target_buffer_idx, uint32_t *out_feat_count);

// Run a transfer command to retrieve the SIFT buffer features from the GPU (run on the asynchronous transfer queue if available)
bool vksift_Memory_copyBufferFeaturesFromGPU(vksift_SiftMemory memory, const uint32_t target_buffer_idx, vksift_Feature *out_features_ptr);

// Run a transfer command to transfer user SIFT features to the SIFT buffer on the GPU (run on the asynchronous transfer queue if available)
bool vksift_Memory_copyBufferFeaturesToGPU(vksift_SiftMemory memory, const uint32_t target_buffer_idx, const vksift_Feature *in_features_ptr,
                                           const uint32_t in_feat_count);

// Read from staging buffer memory to retrieve the number of features matches stored in a matches buffer (GPU not involved in this function)
bool vksift_Memory_getBufferMatchesCount(vksift_SiftMemory memory, uint32_t *out_matches_count);

// Run a transfer command to retrieve the SIFT matches from the GPU (GPU not involved in this function)
bool vksift_Memory_copyBufferMatchesFromGPU(vksift_SiftMemory memory, vksift_Match_2NN *out_matches_ptr);

// Transfer one of the pyramid image to the CPU
bool vksift_Memory_copyPyramidImageFromGPU(vksift_SiftMemory memory, const uint8_t octave, const uint8_t scale, const bool is_dog, float *out_image_data);

// Destory every memory object and free stuffs
void vksift_destroySiftMemory(vksift_SiftMemory *memory_ptr);

#endif // VKSIFT_SIFTMEMORY
