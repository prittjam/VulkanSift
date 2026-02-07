#include "vulkansift/vulkansift.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static uint8_t* load_pgm(const char* path, uint32_t* width, uint32_t* height) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }
    char magic[3];
    if (fscanf(f, "%2s", magic) != 1 || strcmp(magic, "P5") != 0) {
        fprintf(stderr, "Not a PGM file\n"); fclose(f); return NULL;
    }
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
    fgetc(f);
    uint8_t* data = (uint8_t*)malloc(w * h);
    if (fread(data, 1, w * h, f) != w * h) {
        fprintf(stderr, "Short read\n"); free(data); fclose(f); return NULL;
    }
    fclose(f);
    *width = w; *height = h;
    return data;
}

static double get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s image.pgm [2d] [N_iterations]\n", argv[0]);
        return 1;
    }

    bool use_2d = false;
    int n_iter = 20;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "2d") == 0) use_2d = true;
        else n_iter = atoi(argv[i]);
    }

    uint32_t width, height;
    uint8_t* image = load_pgm(argv[1], &width, &height);
    if (!image) return 1;
    printf("Image: %ux%u\n", width, height);

    vksift_Result res = vksift_loadVulkan();
    if (res != VKSIFT_SUCCESS) { free(image); return 1; }

    vksift_setLogLevel(VKSIFT_NO_LOG);

    vksift_Config config = vksift_getDefaultConfig();
    config.input_image_max_size = width * height;
    config.detection_only = true;
    config.use_2d_nms = use_2d;
    config.use_input_upsampling = false;

    vksift_Instance instance = NULL;
    res = vksift_createInstance(&instance, &config);
    if (res != VKSIFT_SUCCESS) { free(image); vksift_unloadVulkan(); return 1; }

    printf("NMS mode: %s, iterations: %d\n", use_2d ? "2D" : "3D", n_iter);

    // Warmup
    vksift_detectFeatures(instance, image, width, height, 0);
    uint32_t nb_feats = vksift_getFeaturesNumber(instance, 0);
    printf("Features: %u (warmup)\n", nb_feats);

    // Benchmark
    double total_detect = 0, total_download = 0;
    for (int i = 0; i < n_iter; i++) {
        double t0 = get_time_ms();
        vksift_detectFeatures(instance, image, width, height, 0);
        nb_feats = vksift_getFeaturesNumber(instance, 0);  // blocks until done
        double t1 = get_time_ms();

        vksift_Feature* feats = malloc(sizeof(vksift_Feature) * nb_feats);
        vksift_downloadFeatures(instance, feats, 0);
        double t2 = get_time_ms();

        total_detect += (t1 - t0);
        total_download += (t2 - t1);
        free(feats);
    }

    printf("\nResults over %d iterations:\n", n_iter);
    printf("  Detect:   %.2f ms avg (%.2f ms total)\n", total_detect / n_iter, total_detect);
    printf("  Download: %.2f ms avg (%.2f ms total)\n", total_download / n_iter, total_download);
    printf("  Total:    %.2f ms avg\n", (total_detect + total_download) / n_iter);
    printf("  Features: %u\n", nb_feats);

    vksift_destroyInstance(&instance);
    vksift_unloadVulkan();
    free(image);
    return 0;
}
