#include "vksift_jl.h"
#include "vulkansift/vulkansift.h"
#include <stdlib.h>
#include <string.h>

struct vksift_jl_context {
    vksift_Instance instance;
    uint32_t last_nb_features;
    int use_rgba_input;
    int use_rgb_input;
};

static void noop_error_handler(vksift_Result err) {
    (void)err;
    // Don't abort — let Julia handle errors
}

vksift_jl_handle vksift_jl_init(
    uint32_t max_width, uint32_t max_height,
    float intensity_threshold, float edge_threshold,
    float seed_scale_sigma, float input_blur_level,
    uint8_t nb_scales_per_octave, uint8_t nb_octaves,
    int use_upsampling, int use_2d_nms,
    int use_rgba_input, int use_rgb_input)
{
    vksift_Result res = vksift_loadVulkan();
    if (res != VKSIFT_SUCCESS) return NULL;

    vksift_setLogLevel(VKSIFT_LOG_WARNING);

    vksift_Config config = vksift_getDefaultConfig();
    config.input_image_max_size = max_width * max_height;
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
    return h;
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

void vksift_jl_destroy(vksift_jl_handle h)
{
    if (!h) return;
    vksift_destroyInstance(&h->instance);
    vksift_unloadVulkan();
    free(h);
}
