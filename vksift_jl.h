#ifndef VKSIFT_JL_H
#define VKSIFT_JL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct vksift_jl_context* vksift_jl_handle;

// Per-feature result (no descriptor, no orientation — detection only)
typedef struct {
    float x;
    float y;
    float sigma;
    int32_t octave_idx;
    uint32_t scale_idx;
    float intensity;  // peak score (DoG value)
} vksift_jl_feature;

// Initialize VulkanSift with detection-only, 2D NMS configuration.
// Returns NULL on failure.
vksift_jl_handle vksift_jl_init(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input);

// Detect features in a grayscale uint8 image (row-major).
// Returns number of detected features.
uint32_t vksift_jl_detect(vksift_jl_handle h,
    const uint8_t* image, uint32_t width, uint32_t height);

// Detect features from an RGB uint8 image (row-major, 3 bytes per pixel).
// Converts to grayscale on the fly before detection.
uint32_t vksift_jl_detect_rgb(vksift_jl_handle h,
    const uint8_t* rgb_image, uint32_t width, uint32_t height);

// Detect features from an RGBA uint8 image (row-major, 4 bytes per pixel).
// Converts to grayscale on the fly before detection (alpha ignored).
uint32_t vksift_jl_detect_rgba(vksift_jl_handle h,
    const uint8_t* rgba_image, uint32_t width, uint32_t height);

// Copy detected features into caller-allocated buffer.
// Buffer must hold at least n features (from vksift_jl_detect return value).
void vksift_jl_get_features(vksift_jl_handle h,
    vksift_jl_feature* out, uint32_t n);

// Destroy context and free GPU resources.
void vksift_jl_destroy(vksift_jl_handle h);

#ifdef __cplusplus
}
#endif

#endif // VKSIFT_JL_H
