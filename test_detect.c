#include "vulkansift/vulkansift.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Simple PGM (grayscale) image loader
static uint8_t* load_pgm(const char* path, uint32_t* width, uint32_t* height) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Not a PGM file\n"); fclose(f); return NULL;
    }

    // Skip comments
    int c = fgetc(f);
    while (c == '#' || c == '\n' || c == ' ') {
        if (c == '#') while (fgetc(f) != '\n');
        c = fgetc(f);
    }
    ungetc(c, f);

    uint32_t w, h, maxval;
    if (fscanf(f, "%u %u %u", &w, &h, &maxval) != 3) {
        fprintf(stderr, "Bad PGM header\n"); fclose(f); return NULL;
    }
    fgetc(f); // consume newline after header

    uint8_t* data = (uint8_t*)malloc(w * h);
    if (fread(data, 1, w * h, f) != w * h) {
        fprintf(stderr, "Short read\n"); free(data); fclose(f); return NULL;
    }
    fclose(f);
    *width = w;
    *height = h;
    return data;
}

static void error_handler(vksift_Result err) {
    fprintf(stderr, "VulkanSift error: %d\n", err);
    // Don't abort - let us handle it
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.pgm [2d]\n", argv[0]);
        return 1;
    }

    bool use_2d = (argc >= 3 && strcmp(argv[2], "2d") == 0);

    uint32_t width, height;
    uint8_t* image = load_pgm(argv[1], &width, &height);
    if (!image) return 1;
    printf("Loaded %ux%u image\n", width, height);

    // Init Vulkan
    vksift_Result res = vksift_loadVulkan();
    if (res != VKSIFT_SUCCESS) {
        fprintf(stderr, "Failed to load Vulkan\n");
        free(image);
        return 1;
    }

    // List GPUs
    uint32_t gpu_count = 0;
    vksift_getAvailableGPUs(&gpu_count, NULL);
    printf("Found %u GPU(s)\n", gpu_count);
    if (gpu_count > 0) {
        VKSIFT_GPU_NAME* names = malloc(sizeof(VKSIFT_GPU_NAME) * gpu_count);
        vksift_getAvailableGPUs(&gpu_count, names);
        for (uint32_t i = 0; i < gpu_count; i++) {
            printf("  GPU %u: %s\n", i, names[i]);
        }
        free(names);
    }

    // Configure
    vksift_Config config = vksift_getDefaultConfig();
    config.input_image_max_size = width * height;
    config.detection_only = true;
    config.use_2d_nms = use_2d;
    config.use_input_upsampling = false; // Match our pipeline: no upsampling
    config.on_error_callback_function = error_handler;

    printf("Mode: detection_only=%d, use_2d_nms=%d\n", config.detection_only, config.use_2d_nms);

    // Create instance
    vksift_Instance instance = NULL;
    res = vksift_createInstance(&instance, &config);
    if (res != VKSIFT_SUCCESS) {
        fprintf(stderr, "Failed to create VulkanSift instance\n");
        free(image);
        vksift_unloadVulkan();
        return 1;
    }

    // Detect
    printf("Detecting features...\n");
    vksift_detectFeatures(instance, image, width, height, 0);

    // Download results
    uint32_t nb_feats = vksift_getFeaturesNumber(instance, 0);
    printf("Detected %u features\n", nb_feats);

    if (nb_feats > 0 && nb_feats < 1000000) {
        vksift_Feature* feats = malloc(sizeof(vksift_Feature) * nb_feats);
        vksift_downloadFeatures(instance, feats, 0);

        // Print first 10
        uint32_t n = nb_feats < 10 ? nb_feats : 10;
        printf("\nFirst %u features:\n", n);
        for (uint32_t i = 0; i < n; i++) {
            printf("  [%u] x=%.1f y=%.1f sigma=%.3f octave=%d scale=%u intensity=%.6f\n",
                   i, feats[i].x, feats[i].y, feats[i].sigma,
                   feats[i].octave_idx, feats[i].scale_idx, feats[i].intensity);
        }
        free(feats);
    }

    // Cleanup
    vksift_destroyInstance(&instance);
    vksift_unloadVulkan();
    free(image);

    printf("Done.\n");
    return 0;
}
