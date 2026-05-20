#include "vksift_jl.h"
#include "vulkansift/vulkansift.h"
#include <stdlib.h>
#include <string.h>

struct vksift_jl_context {
    vksift_Instance instance;
    uint32_t last_nb_features;
    int use_rgba_input;
    int use_rgb_input;
    uint32_t nb_pyramid_slots;  // cached at init so dispatch_parallel_imas can chunk correctly

    // Phase C-3 parallel-dispatch state. Per-warp feature cache populated by
    // vksift_jl_dispatch_parallel_imas at the end of each wave (before the
    // next wave overwrites the slot's SIFT buffer). Caller reads back via
    // vksift_jl_get_features_for_warp.
    //
    // Layout: parallel_features[warp_idx] is a malloc'd vksift_jl_feature
    // array of length parallel_counts[warp_idx]. Both arrays have
    // parallel_n_warps entries. NULL/0 when no parallel dispatch has been
    // called yet.
    vksift_jl_feature **parallel_features;
    uint32_t          *parallel_counts;
    uint32_t           parallel_n_warps;
};

static void noop_error_handler(vksift_Result err) {
    (void)err;
    // Don't abort — let Julia handle errors
}

vksift_jl_handle vksift_jl_init_ex(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input,
    uint32_t nb_pyramid_slots)
{
    vksift_Result res = vksift_loadVulkan();
    if (res != VKSIFT_SUCCESS) return NULL;

    vksift_setLogLevel(VKSIFT_LOG_WARNING);

    vksift_Config config = vksift_getDefaultConfig();
    // Pass non-square explicit dims through so VKS allocates a max_width × max_height
    // canvas rather than a square ceil(sqrt(W·H)) one.
    config.input_image_max_size = max_width * max_height;
    config.input_image_max_width = max_width;
    config.input_image_max_height = max_height;
    config.intensity_threshold = intensity_threshold;
    config.edge_threshold = edge_threshold;
    config.seed_scale_sigma = seed_scale_sigma;
    config.input_image_blur_level = input_blur_level;
    config.nb_scales_per_octave = nb_scales_per_octave;
    config.nb_octaves = nb_octaves;
    config.use_input_upsampling = use_upsampling ? true : false;
    config.detection_only = true;
    config.use_2d_nms = use_2d_nms ? true : false;
    config.use_rgba_input = use_rgba_input ? true : false;
    config.use_rgb_input = use_rgb_input ? true : false;
    config.on_error_callback_function = noop_error_handler;
    config.nb_pyramid_slots = nb_pyramid_slots == 0u ? 1u : nb_pyramid_slots;

    vksift_Instance instance = NULL;
    res = vksift_createInstance(&instance, &config);
    if (res != VKSIFT_SUCCESS) {
        vksift_unloadVulkan();
        return NULL;
    }

    vksift_jl_handle h = (vksift_jl_handle)malloc(sizeof(struct vksift_jl_context));
    h->instance = instance;
    h->last_nb_features = 0;
    h->use_rgba_input = use_rgba_input;
    h->use_rgb_input = use_rgb_input;
    h->parallel_features = NULL;
    h->parallel_counts = NULL;
    h->parallel_n_warps = 0u;
    h->nb_pyramid_slots = nb_pyramid_slots == 0u ? 1u : nb_pyramid_slots;
    return h;
}

vksift_jl_handle vksift_jl_init(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input)
{
    return vksift_jl_init_ex(max_width, max_height,
                             intensity_threshold, edge_threshold,
                             seed_scale_sigma, input_blur_level,
                             nb_scales_per_octave, nb_octaves,
                             use_upsampling, use_2d_nms,
                             use_rgba_input, use_rgb_input,
                             1u);
}

void vksift_jl_set_affine_warp(vksift_jl_handle h,
    float a11, float a12, float a13,
    float a21, float a22, float a23,
    float fill_value)
{
    if (!h) return;
    vksift_setPendingAffineWarpInstance(h->instance, a11, a12, a13, a21, a22, a23, fill_value);
}

void vksift_jl_set_preblur(vksift_jl_handle h,
    float sigma, float dir_x, float dir_y)
{
    if (!h) return;
    vksift_setPendingPreBlurInstance(h->instance, sigma, dir_x, dir_y);
}

uint32_t vksift_jl_detect(vksift_jl_handle h,
    const uint8_t* image, uint32_t width, uint32_t height)
{
    vksift_detectFeatures(h->instance, image, width, height, 0);
    h->last_nb_features = vksift_getFeaturesNumber(h->instance, 0);
    return h->last_nb_features;
}

uint32_t vksift_jl_detect_rgb(vksift_jl_handle h,
    const uint8_t* rgb_image, uint32_t width, uint32_t height)
{
    if (h->use_rgb_input) {
        // GPU RGB→Gray path: pass RGB data directly
        vksift_detectFeatures(h->instance, rgb_image, width, height, 0);
    } else {
        // CPU fallback: convert RGB→Gray on CPU
        uint32_t npixels = width * height;
        uint8_t* gray = (uint8_t*)malloc(npixels);
        for (uint32_t i = 0; i < npixels; i++) {
            uint32_t r = rgb_image[i * 3];
            uint32_t g = rgb_image[i * 3 + 1];
            uint32_t b = rgb_image[i * 3 + 2];
            gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        }
        vksift_detectFeatures(h->instance, gray, width, height, 0);
        free(gray);
    }
    h->last_nb_features = vksift_getFeaturesNumber(h->instance, 0);
    return h->last_nb_features;
}

uint32_t vksift_jl_detect_rgba(vksift_jl_handle h,
    const uint8_t* rgba_image, uint32_t width, uint32_t height)
{
    if (h->use_rgba_input) {
        // GPU RGBA→Gray path: pass RGBA data directly
        vksift_detectFeatures(h->instance, rgba_image, width, height, 0);
    } else {
        // CPU fallback: convert RGBA→Gray on CPU
        uint32_t npixels = width * height;
        uint8_t* gray = (uint8_t*)malloc(npixels);
        for (uint32_t i = 0; i < npixels; i++) {
            uint32_t r = rgba_image[i * 4];
            uint32_t g = rgba_image[i * 4 + 1];
            uint32_t b = rgba_image[i * 4 + 2];
            gray[i] = (uint8_t)((r * 77 + g * 150 + b * 29) >> 8);
        }
        vksift_detectFeatures(h->instance, gray, width, height, 0);
        free(gray);
    }
    h->last_nb_features = vksift_getFeaturesNumber(h->instance, 0);
    return h->last_nb_features;
}

void vksift_jl_get_features(vksift_jl_handle h,
    vksift_jl_feature* out, uint32_t n)
{
    if (n == 0) return;

    vksift_Feature* feats = (vksift_Feature*)malloc(sizeof(vksift_Feature) * n);
    vksift_downloadFeatures(h->instance, feats, 0);

    for (uint32_t i = 0; i < n; i++) {
        out[i].x = feats[i].x;
        out[i].y = feats[i].y;
        out[i].sigma = feats[i].sigma;
        out[i].octave_idx = feats[i].octave_idx;
        out[i].scale_idx = feats[i].scale_idx;
        out[i].intensity = feats[i].intensity;
    }
    free(feats);
}

static void free_parallel_cache(vksift_jl_handle h)
{
    if (!h) return;
    if (h->parallel_features) {
        for (uint32_t w = 0u; w < h->parallel_n_warps; ++w) {
            free(h->parallel_features[w]);
        }
        free(h->parallel_features);
        h->parallel_features = NULL;
    }
    if (h->parallel_counts) {
        free(h->parallel_counts);
        h->parallel_counts = NULL;
    }
    h->parallel_n_warps = 0u;
}

void vksift_jl_destroy(vksift_jl_handle h)
{
    if (!h) return;
    free_parallel_cache(h);
    vksift_destroyInstance(&h->instance);
    vksift_unloadVulkan();
    free(h);
}

// ============================================================================
// IMAS (ASIFT) tilt-warp entrypoint. Caller must have uploaded the source
// image via vksift_jl_detect() first (any peak threshold; detection result
// discarded). Then calls:
//   vksift_jl_run_imas(h, W, H, t, phi, &out_w, &out_h)
// which runs the 5-shader GPU chain and writes the tilted Float32 result to
// a host-mapped buffer. Caller retrieves the buffer via
//   vksift_jl_get_imas_buffer(h)
// (must be read before the next vksift_jl_run_imas / vksift_jl_detect call).
// ============================================================================

int vksift_jl_run_imas(vksift_jl_handle h, uint32_t W, uint32_t H,
                       float t_factor, float theta_rad,
                       uint32_t *out_w, uint32_t *out_h)
{
    if (!h) return 0;
    return vksift_runImas(h->instance, W, H, t_factor, theta_rad, out_w, out_h) ? 1 : 0;
}

const float *vksift_jl_get_imas_buffer(vksift_jl_handle h)
{
    if (!h) return NULL;
    return vksift_getImasReadbackPtr(h->instance);
}

uint32_t vksift_jl_detect_on_imas(vksift_jl_handle h,
                                  uint32_t canvas_w, uint32_t canvas_h,
                                  uint32_t valid_w,  uint32_t valid_h,
                                  float    fill_value)
{
    if (!h) return 0;
    vksift_detectFeaturesOnImas(h->instance, canvas_w, canvas_h, valid_w, valid_h, fill_value, 0);
    h->last_nb_features = vksift_getFeaturesNumber(h->instance, 0);
    return h->last_nb_features;
}

// Phase B-3 FFI — fused IMAS + Quantize + SIFT-detect, single GPU submission.
// Mirrors vksift_jl_detect_on_imas's pattern: wraps the public C entrypoint
// and returns the feature count for the caller to allocate the download
// buffer. See vksift_jl.h for usage.
uint32_t vksift_jl_detect_fused_imas(vksift_jl_handle h,
                                     uint32_t W, uint32_t H,
                                     float t_factor, float theta_rad,
                                     uint32_t canvas_w, uint32_t canvas_h)
{
    if (!h) return 0;
    vksift_detectFeaturesFusedImas(h->instance, W, H, t_factor, theta_rad,
                                   canvas_w, canvas_h, 0);
    h->last_nb_features = vksift_getFeaturesNumber(h->instance, 0);
    return h->last_nb_features;
}

// Phase C-3 FFI — parallel-pyramid IMAS+detect dispatch. Wraps
// vksift_dispatchParallelIMAS at the FFI level so feature downloads for one
// wave happen BEFORE the next wave overwrites sift_buffer_arr[s]. The
// per-warp feature arrays are owned by the handle and freed on the next
// dispatch_parallel_imas call (or on destroy).
//
// Implementation: chunk into waves of nb_pyramid_slots, call the C-level
// dispatch with n_warps = wave (single wave per call), then immediately
// download each slot's features into the handle-owned cache.
void vksift_jl_dispatch_parallel_imas(
    vksift_jl_handle h,
    uint32_t W, uint32_t H,
    const float *t_factors, const float *theta_rads, uint32_t n_warps,
    uint32_t canvas_w, uint32_t canvas_h,
    uint32_t *out_features_per_warp)
{
    if (!h || !t_factors || !theta_rads || n_warps == 0u || !out_features_per_warp) return;

    // Free any prior parallel cache + reallocate.
    free_parallel_cache(h);
    h->parallel_n_warps = n_warps;
    h->parallel_features = (vksift_jl_feature **)calloc(n_warps, sizeof(vksift_jl_feature *));
    h->parallel_counts   = (uint32_t *)calloc(n_warps, sizeof(uint32_t));
    if (!h->parallel_features || !h->parallel_counts) {
        free_parallel_cache(h);
        return;
    }

    // Chunk to the instance's actual nb_pyramid_slots so each FFI call to
    // vksift_dispatchParallelIMAS produces exactly ONE internal wave — that
    // way each slot's SIFT buffer holds its warp's features when we download
    // (the next wave would overwrite them). MAX_WAVE = 8 is the static upper
    // bound on nb_pyramid_slots; if the real value is smaller (e.g. 1), we
    // chunk smaller and the FFI's download loop only touches valid slots.
    enum { MAX_WAVE = 8 /* = VKSIFT_MAX_PYRAMID_SLOTS */ };
    const uint32_t wave_size = (h->nb_pyramid_slots == 0u || h->nb_pyramid_slots > MAX_WAVE)
                               ? MAX_WAVE : h->nb_pyramid_slots;

    // Reusable per-wave scratch — sized for the maximum possible wave.
    uint32_t       wave_counts[MAX_WAVE];
    vksift_WarpSpec wave_specs[MAX_WAVE];

    for (uint32_t base = 0u; base < n_warps; ) {
        uint32_t remaining = n_warps - base;
        uint32_t wave = remaining < wave_size ? remaining : wave_size;
        for (uint32_t s = 0u; s < wave; ++s) {
            wave_specs[s].t_factor  = t_factors[base + s];
            wave_specs[s].theta_rad = theta_rads[base + s];
            wave_counts[s] = 0u;
        }
        // The C-level dispatch will internally chunk to min(n_slots, wave).
        // We pass wave ≤ MAX_WAVE = max possible n_slots, so it does exactly
        // one wave per call — feature buffers stay valid until we download.
        vksift_dispatchParallelIMAS(h->instance, W, H,
                                    wave_specs, wave,
                                    canvas_w, canvas_h, wave_counts);

        for (uint32_t s = 0u; s < wave; ++s) {
            uint32_t n = wave_counts[s];
            h->parallel_counts[base + s] = n;
            out_features_per_warp[base + s] = n;
            if (n == 0u) {
                h->parallel_features[base + s] = NULL;
                continue;
            }
            // Download from sift_buffer_arr[s] into a fresh handle-owned
            // buffer. vksift_downloadFeatures waits the detect fence if the
            // GPU is still using the buffer (it isn't — we just submitted
            // and waited).
            vksift_Feature *raw = (vksift_Feature *)malloc(sizeof(vksift_Feature) * n);
            if (!raw) {
                h->parallel_features[base + s] = NULL;
                continue;
            }
            vksift_downloadFeatures(h->instance, raw, s);

            vksift_jl_feature *out = (vksift_jl_feature *)malloc(sizeof(vksift_jl_feature) * n);
            if (!out) {
                free(raw);
                h->parallel_features[base + s] = NULL;
                continue;
            }
            for (uint32_t i = 0u; i < n; ++i) {
                out[i].x         = raw[i].x;
                out[i].y         = raw[i].y;
                out[i].sigma     = raw[i].sigma;
                out[i].octave_idx= raw[i].octave_idx;
                out[i].scale_idx = raw[i].scale_idx;
                out[i].intensity = raw[i].intensity;
            }
            free(raw);
            h->parallel_features[base + s] = out;
        }
        base += wave;
    }
}

void vksift_jl_get_features_for_warp(
    vksift_jl_handle h, uint32_t warp_idx,
    vksift_jl_feature *out, uint32_t n)
{
    if (!h || !out || n == 0u) return;
    if (warp_idx >= h->parallel_n_warps) return;
    vksift_jl_feature *src = h->parallel_features[warp_idx];
    if (!src) return;
    uint32_t avail = h->parallel_counts[warp_idx];
    uint32_t copy_n = (n < avail) ? n : avail;
    memcpy(out, src, sizeof(vksift_jl_feature) * copy_n);
}
