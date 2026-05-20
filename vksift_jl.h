#ifndef VKSIFT_JL_H
#define VKSIFT_JL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle
typedef struct vksift_jl_context* vksift_jl_handle;

// Per-feature result (no descriptor, no orientation — detection only).
//
// DoG sign convention (matches VLFeat / BlobBoards / VGF):
//   DoG[s] = G[s+1] − G[s], so polarity follows the local extremum:
//     - DARK blob (intensity valley): DoG response is a local MAXIMUM
//       in scale space, intensity > 0 here.
//     - LIGHT blob (intensity peak): DoG response is a local MINIMUM,
//       intensity < 0 here.
//   To filter by polarity Julia-side use `f.intensity > 0` for dark
//   blobs (typical SIFT default), `f.intensity < 0` for light blobs.
typedef struct {
    float x;
    float y;
    float sigma;
    int32_t octave_idx;
    uint32_t scale_idx;
    float intensity;  // signed DoG response; sign = polarity (see above)
} vksift_jl_feature;

// Initialize VulkanSift with detection-only, 2D NMS configuration.
// Returns NULL on failure. Defaults to nb_pyramid_slots=1 (single-slot
// semantics). Use vksift_jl_init_ex to request a higher slot count for the
// Phase C-3 parallel IMAS dispatch.
vksift_jl_handle vksift_jl_init(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input);

// Phase C-3: extended init with nb_pyramid_slots. Identical to vksift_jl_init
// except the caller can request a pyramid-slot count > 1 for the parallel
// IMAS path. nb_pyramid_slots is clamped to VKSIFT_MAX_PYRAMID_SLOTS (= 8).
// Pass 1 for the legacy single-slot behavior.
vksift_jl_handle vksift_jl_init_ex(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input,
    uint32_t nb_pyramid_slots);

// Set the pending AffineWarp matrix used by the NEXT detect call.
// 2x3 inverse affine mapping warped pixel → input pixel:
//   col_in = a11*col_out + a12*row_out + a13
//   row_in = a21*col_out + a22*row_out + a23
// fill_value (normalized [0..1]) returned for out-of-bounds samples.
// Identity (a11=a22=1, rest=0, fill=0) leaves the pipeline as-is.
void vksift_jl_set_affine_warp(vksift_jl_handle h,
    float a11, float a12, float a13,
    float a21, float a22, float a23,
    float fill_value);

// Set the σ_aa pre-blur applied to the input image before AffineWarp on the
// NEXT detect call. Direction (dir_x, dir_y) is the 1D blur axis in input
// pixel coords (ASIFT uses (sin φ, cos φ) for the squash direction). σ = 0
// yields a pass-through copy (no blur).
void vksift_jl_set_preblur(vksift_jl_handle h,
    float sigma, float dir_x, float dir_y);

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

// =========================================================================
// IMAS (ASIFT) GPU tilt-warp pipeline (sift_imas.c).
//
// Reproduces fast_imas_IPOL `simulate_digital_tilt` fully on GPU:
//   AffineWarp_rot(input, θ) → GaussBlur1D_v(σ_aa = 0.8·√(t²-1))
//   → finvspline_row → finvspline_col → fproj_cubic_y(t)
// Result: a Float32 tilted image of size W_rot × ⌊H_rot/t⌋ landed in a
// host-mapped readback buffer (`vksift_jl_get_imas_buffer`).
//
// Usage:
//   1. vksift_jl_detect(h, original_bytes, W, H)   // primer: uploads input
//   2. vksift_jl_run_imas(h, W, H, t, theta, &out_w, &out_h)
//   3. const float *tilted = vksift_jl_get_imas_buffer(h);
//      // read out_w × out_h Float32 pixels (row-major)
//   4. (caller converts Float32 → UInt8 and calls vksift_jl_detect again,
//       this time with identity warp + σ_aa=0 to detect on tilted image)
//
// Returns non-zero on success.
// =========================================================================
int vksift_jl_run_imas(vksift_jl_handle h, uint32_t W, uint32_t H,
                       float t_factor, float theta_rad,
                       uint32_t *out_w, uint32_t *out_h);

const float *vksift_jl_get_imas_buffer(vksift_jl_handle h);

// Run SIFT detection on the most-recent IMAS-tilted image (mem->rotated_image,
// device-side) — no host roundtrip. Caller must have run vksift_jl_run_imas()
// first.
//
// (canvas_w, canvas_h): SIFT pyramid input dims — pass the SAME values for
//   every warp in an IMAS schedule (e.g. max tilted W×H), so the pyramid
//   stays stable and isn't reallocated per warp.
// (valid_w, valid_h)  : actual dims the IMAS pipeline wrote (= out_w/out_h
//   from vksift_jl_run_imas). The quantize shader writes those pixels from
//   rotated_image and fills the rest of input_image with `fill_value` (a
//   normalized float in [0, 1]; typically 0.5).
//
// Returns the number of features detected; use vksift_jl_get_features().
uint32_t vksift_jl_detect_on_imas(vksift_jl_handle h,
                                  uint32_t canvas_w, uint32_t canvas_h,
                                  uint32_t valid_w,  uint32_t valid_h,
                                  float    fill_value);

// Phase B-3: fully-fused IMAS + Quantize + SIFT-detect in a single GPU
// submission. Combines what vksift_jl_run_imas + vksift_jl_detect_on_imas did
// separately into one pre-recorded command buffer per slot — one queue submit,
// one fence wait, no host roundtrip.
//
// Caller must have uploaded the original (untilted) image via
// vksift_jl_detect() first to populate cached_input_image. After that,
// vksift_jl_detect_fused_imas builds the per-warp IMAS params (rotated canvas,
// affine matrix, sigma_aa, dispatch counts) and submits the fused chain.
//
// (W, H)             : input-image dims used at upload.
// (t_factor, theta)  : IMAS warp params.
// (canvas_w, canvas_h): SIFT pyramid canvas — keep identical across warps.
// Returns the number of features; use vksift_jl_get_features().
uint32_t vksift_jl_detect_fused_imas(vksift_jl_handle h,
                                     uint32_t W, uint32_t H,
                                     float t_factor, float theta_rad,
                                     uint32_t canvas_w, uint32_t canvas_h);

// Phase C-3: parallel-pyramid IMAS+detect dispatch. Submits n_warps warps in
// waves of size min(nb_pyramid_slots, remaining). After EACH wave (and
// BEFORE the next wave overwrites sift_buffer_arr[s]), the FFI downloads the
// wave's features into a caller-invisible host-side cache keyed by warp_idx.
// This makes the per-warp data safe to read until the next call to
// vksift_jl_dispatch_parallel_imas (or vksift_jl_destroy).
//
// Caller must have uploaded the original (untilted) image via
// vksift_jl_detect() first to populate cached_input_image.
//
// out_features_per_warp must be sized at least n_warps. On return, it holds
// the number of features detected for each warp.
void vksift_jl_dispatch_parallel_imas(
    vksift_jl_handle h,
    uint32_t W, uint32_t H,
    const float *t_factors, const float *theta_rads, uint32_t n_warps,
    uint32_t canvas_w, uint32_t canvas_h,
    uint32_t *out_features_per_warp);

// Phase C-3: read features for warp `warp_idx` from the most recent
// vksift_jl_dispatch_parallel_imas call. The caller must allocate `out` with
// at least `n` entries, where n ≤ out_features_per_warp[warp_idx]. Features
// come from the FFI's host-side cache; safe to call any number of times until
// the next dispatch_parallel_imas / destroy.
void vksift_jl_get_features_for_warp(
    vksift_jl_handle h, uint32_t warp_idx,
    vksift_jl_feature *out, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif // VKSIFT_JL_H
