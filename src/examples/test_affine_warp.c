// test_affine_warp.c
//
// Standalone pixel-accuracy test for shaders/AffineWarp.comp.
//
// Generates a synthetic gradient pattern, runs AffineWarp on the GPU with a
// known affine matrix, then compares the result against a CPU bilinear
// reference implementing the same map. Passes if max |error| < 1e-4 over
// all output pixels.
//
// Two cases are exercised:
//   1. Pure integer translation — output must be exactly equal to the
//      shifted input (no interpolation).
//   2. Anisotropic downscale by 1.5x in each axis — exercises bilinear
//      sampling and out-of-bounds handling.

#include "vulkansift/vkenv/vulkan_device.h"
#include "vulkansift/vkenv/vulkan_utils.h"
#include "vulkansift/vkenv/logger.h"

#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IN_W  32
#define IN_H  32
#define OUT_W 32
#define OUT_H 32
#define WG    8

static const char *LOG_TAG = "AffineWarpTest";

// Match the GLSL push_constant layout in AffineWarp.comp exactly.
typedef struct {
    uint32_t output_width;
    uint32_t output_height;
    float    a11, a12, a13;
    float    a21, a22, a23;
    float    fill_value;
} PushConst;

// Generate a deterministic non-trivial input pattern.
static void generate_input(float *buf) {
    for (int y = 0; y < IN_H; y++) {
        for (int x = 0; x < IN_W; x++) {
            buf[y * IN_W + x] = (float)(x * 7 + y * 3) / 255.0f;
        }
    }
}

// CPU bilinear reference matching the shader: sample input at the inverse-affine
// position, with the same +0.5 pixel-center texture convention used in GLSL.
static float cpu_bilinear(const float *in, int W, int H, float x_in, float y_in,
                          float fill_value) {
    // Texture coord match: tex_x = (x_in + 0.5) / W
    // Bilinear means we sample at x_in (continuous pixel-center coord),
    // interpolating between pixels (floor, floor+1).
    if (x_in < -0.5f || x_in > (float)W - 0.5f ||
        y_in < -0.5f || y_in > (float)H - 0.5f) {
        return fill_value;
    }
    // Sample at pixel-center convention: nearest pixel = round
    float xf = x_in;
    float yf = y_in;
    int x0 = (int)floorf(xf);
    int y0 = (int)floorf(yf);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    float ax = xf - (float)x0;
    float ay = yf - (float)y0;
    // Clamp to edge for sampling (matches sampler clamp-to-edge mode)
    int x0c = (x0 < 0) ? 0 : (x0 >= W ? W - 1 : x0);
    int x1c = (x1 < 0) ? 0 : (x1 >= W ? W - 1 : x1);
    int y0c = (y0 < 0) ? 0 : (y0 >= H ? H - 1 : y0);
    int y1c = (y1 < 0) ? 0 : (y1 >= H ? H - 1 : y1);
    float v00 = in[y0c * W + x0c];
    float v10 = in[y0c * W + x1c];
    float v01 = in[y1c * W + x0c];
    float v11 = in[y1c * W + x1c];
    return (1 - ax) * (1 - ay) * v00 + ax * (1 - ay) * v10 +
           (1 - ax) * ay * v01 + ax * ay * v11;
}

static void cpu_reference(const float *in, float *out, PushConst pc) {
    for (uint32_t y = 0; y < pc.output_height; y++) {
        for (uint32_t x = 0; x < pc.output_width; x++) {
            float x_in = pc.a11 * (float)x + pc.a12 * (float)y + pc.a13;
            float y_in = pc.a21 * (float)x + pc.a22 * (float)y + pc.a23;
            out[y * pc.output_width + x] =
                cpu_bilinear(in, IN_W, IN_H, x_in, y_in, pc.fill_value);
        }
    }
}

// Helper: allocate device memory matched to image's requirements.
static bool alloc_and_bind_image_memory(vkenv_Device dev, VkImage img,
                                        VkDeviceMemory *mem,
                                        VkMemoryPropertyFlags props) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev->device, img, &req);
    uint32_t mem_idx;
    if (!vkenv_findValidMemoryType(dev->physical_device, req, props, &mem_idx)) {
        logError(LOG_TAG, "no compatible memory type for image");
        return false;
    }
    if (!vkenv_allocateMemory(mem, dev, req.size, mem_idx)) {
        logError(LOG_TAG, "failed to allocate image memory");
        return false;
    }
    if (!vkenv_bindImageMemory(dev, img, *mem, 0)) {
        logError(LOG_TAG, "failed to bind image memory");
        return false;
    }
    return true;
}

static bool alloc_and_bind_buffer_memory(vkenv_Device dev, VkBuffer buf,
                                         VkDeviceMemory *mem,
                                         VkMemoryPropertyFlags props) {
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev->device, buf, &req);
    uint32_t mem_idx;
    if (!vkenv_findValidMemoryType(dev->physical_device, req, props, &mem_idx)) {
        return false;
    }
    if (!vkenv_allocateMemory(mem, dev, req.size, mem_idx)) return false;
    if (!vkenv_bindBufferMemory(dev, buf, *mem, 0)) return false;
    return true;
}

// Run one warp test case. Returns max abs error, or -1 on failure.
static float run_test_case(vkenv_Device dev, VkCommandPool cmd_pool,
                           VkDescriptorPool desc_pool,
                           VkDescriptorSetLayout dsl, VkPipelineLayout playout,
                           VkPipeline pipeline, VkSampler sampler,
                           VkImageView in_view, VkImageView out_view,
                           VkImage in_image, VkImage out_image,
                           VkBuffer download_buf,
                           VkDeviceMemory download_mem,
                           PushConst pc, const float *in_data,
                           const char *case_name) {
    // Allocate descriptor set
    VkDescriptorSet desc_set;
    VkDescriptorSetAllocateInfo dsai = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = desc_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &dsl,
    };
    if (vkAllocateDescriptorSets(dev->device, &dsai, &desc_set) != VK_SUCCESS) {
        logError(LOG_TAG, "vkAllocateDescriptorSets failed"); return -1.0f;
    }
    VkDescriptorImageInfo in_info = {
        .sampler = sampler, .imageView = in_view,
        .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    };
    VkDescriptorImageInfo out_info = {
        .sampler = VK_NULL_HANDLE, .imageView = out_view,
        .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
    };
    VkWriteDescriptorSet writes[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .pImageInfo = &in_info},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = desc_set,
         .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .pImageInfo = &out_info},
    };
    vkUpdateDescriptorSets(dev->device, 2, writes, 0, NULL);

    // Record + submit one-shot command buffer: transition output to GENERAL,
    // dispatch shader, transition output to TRANSFER_SRC, copy to download buf.
    VkCommandBuffer cb;
    if (!vkenv_beginInstantCommandBuffer(dev->device, cmd_pool, &cb)) return -1.0f;

    VkImageMemoryBarrier bar_out_to_gen = vkenv_genImageMemoryBarrier(
        out_image, 0, VK_ACCESS_SHADER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, NULL, 0, NULL, 1, &bar_out_to_gen);

    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, playout,
                            0, 1, &desc_set, 0, NULL);
    vkCmdPushConstants(cb, playout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(PushConst), &pc);
    uint32_t nx = (pc.output_width  + WG - 1) / WG;
    uint32_t ny = (pc.output_height + WG - 1) / WG;
    vkCmdDispatch(cb, nx, ny, 1);

    VkImageMemoryBarrier bar_out_to_xfer = vkenv_genImageMemoryBarrier(
        out_image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 0, NULL, 0, NULL, 1, &bar_out_to_xfer);

    VkBufferImageCopy copy = {
        .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {pc.output_width, pc.output_height, 1},
    };
    vkCmdCopyImageToBuffer(cb, out_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           download_buf, 1, &copy);

    if (!vkenv_endInstantCommandBuffer(dev->device, dev->general_queues[0],
                                        cmd_pool, cb)) return -1.0f;

    // Reset output image layout for next use (will be transitioned UNDEFINED→GENERAL again)
    // (no explicit action needed; next case starts with UNDEFINED layout)
    (void)in_data;
    (void)in_image;

    // Map download buffer, compare to CPU reference
    void *mapped;
    size_t out_bytes = pc.output_width * pc.output_height * sizeof(float);
    if (vkMapMemory(dev->device, download_mem, 0, out_bytes, 0, &mapped) != VK_SUCCESS) {
        return -1.0f;
    }
    float *gpu_out = (float*)mapped;
    float cpu_out[OUT_W * OUT_H];
    cpu_reference(in_data, cpu_out, pc);
    float max_err = 0.f, sum_err = 0.f;
    int n = pc.output_width * pc.output_height;
    for (int i = 0; i < n; i++) {
        float e = fabsf(gpu_out[i] - cpu_out[i]);
        if (e > max_err) max_err = e;
        sum_err += e;
    }
    printf("  [%s]  max_err=%.6f  mean_err=%.6f\n",
           case_name, max_err, sum_err / (float)n);
    vkUnmapMemory(dev->device, download_mem);

    vkFreeDescriptorSets(dev->device, desc_pool, 1, &desc_set);
    return max_err;
}

int main(void) {
    vkenv_setLogLevel(VKENV_LOG_INFO);
    // 1. Instance + device
    vkenv_InstanceConfig icfg = {
        .application_name = "AffineWarpTest", .application_version = 1,
        .engine_name = "vkenv", .engine_version = 1,
        .vulkan_api_version = VK_API_VERSION_1_2,
        .validation_layer_count = 0, .validation_layers = NULL,
        .instance_extension_count = 0, .instance_extensions = NULL,
    };
    if (!vkenv_createInstance(&icfg)) {
        fprintf(stderr, "vkenv_createInstance failed\n"); return 1;
    }
    vkenv_DeviceConfig dcfg = {
        .device_extension_count = 0, .device_extensions = NULL,
        .nb_general_queues = 1, .nb_async_compute_queues = 0,
        .nb_async_transfer_queues = 0, .target_device_idx = -1,
    };
    vkenv_Device dev = NULL;
    if (!vkenv_createDevice(&dev, &dcfg)) {
        fprintf(stderr, "vkenv_createDevice failed\n");
        vkenv_destroyInstance(); return 1;
    }
    printf("Device: %s\n", dev->physical_device_props.deviceName);

    // 2. Command pool
    VkCommandPool cmd_pool;
    VkCommandPoolCreateInfo cpci = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = dev->general_queues_family_idx,
    };
    if (vkCreateCommandPool(dev->device, &cpci, NULL, &cmd_pool) != VK_SUCCESS) {
        return 1;
    }

    // 3. Input + output images (r32f)
    VkImage in_image, out_image;
    VkDeviceMemory in_mem, out_mem;
    vkenv_createImage(&in_image, dev, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT,
                      (VkExtent3D){IN_W, IN_H, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                      VK_IMAGE_TILING_OPTIMAL,
                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                      VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED);
    alloc_and_bind_image_memory(dev, in_image, &in_mem,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    vkenv_createImage(&out_image, dev, 0, VK_IMAGE_TYPE_2D, VK_FORMAT_R32_SFLOAT,
                      (VkExtent3D){OUT_W, OUT_H, 1}, 1, 1, VK_SAMPLE_COUNT_1_BIT,
                      VK_IMAGE_TILING_OPTIMAL,
                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                      VK_SHARING_MODE_EXCLUSIVE, 0, NULL, VK_IMAGE_LAYOUT_UNDEFINED);
    alloc_and_bind_image_memory(dev, out_image, &out_mem,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    VkImageView in_view, out_view;
    vkenv_createImageView(&in_view, dev, 0, in_image, VK_IMAGE_VIEW_TYPE_2D,
                          VK_FORMAT_R32_SFLOAT, VKENV_DEFAULT_COMPONENT_MAPPING,
                          (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,
                                                    0, 1, 0, 1});
    vkenv_createImageView(&out_view, dev, 0, out_image, VK_IMAGE_VIEW_TYPE_2D,
                          VK_FORMAT_R32_SFLOAT, VKENV_DEFAULT_COMPONENT_MAPPING,
                          (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT,
                                                    0, 1, 0, 1});

    // 4. Sampler — linear, clamp-to-edge
    VkSamplerCreateInfo sci = {
        .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
        .unnormalizedCoordinates = VK_FALSE,
    };
    VkSampler sampler;
    vkCreateSampler(dev->device, &sci, NULL, &sampler);

    // 5. Upload input data via staging buffer
    size_t in_bytes = IN_W * IN_H * sizeof(float);
    VkBuffer upload_buf;
    VkDeviceMemory upload_mem;
    vkenv_createBuffer(&upload_buf, dev, 0, in_bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                       VK_SHARING_MODE_EXCLUSIVE, 0, NULL);
    alloc_and_bind_buffer_memory(dev, upload_buf, &upload_mem,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    float *in_data = (float*)malloc(in_bytes);
    generate_input(in_data);
    void *mapped;
    vkMapMemory(dev->device, upload_mem, 0, in_bytes, 0, &mapped);
    memcpy(mapped, in_data, in_bytes);
    vkUnmapMemory(dev->device, upload_mem);
    // Copy upload → input image; transition in_image → SHADER_READ_ONLY_OPTIMAL
    VkCommandBuffer setup_cb;
    vkenv_beginInstantCommandBuffer(dev->device, cmd_pool, &setup_cb);
    VkImageMemoryBarrier bar_in_to_dst = vkenv_genImageMemoryBarrier(
        in_image, 0, VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(setup_cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL,
                         1, &bar_in_to_dst);
    VkBufferImageCopy bic = {
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageExtent = {IN_W, IN_H, 1},
    };
    vkCmdCopyBufferToImage(setup_cb, upload_buf, in_image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bic);
    VkImageMemoryBarrier bar_in_to_read = vkenv_genImageMemoryBarrier(
        in_image, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(setup_cb, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL,
                         0, NULL, 1, &bar_in_to_read);
    vkenv_endInstantCommandBuffer(dev->device, dev->general_queues[0],
                                   cmd_pool, setup_cb);

    // 6. Download buffer
    size_t out_bytes = OUT_W * OUT_H * sizeof(float);
    VkBuffer download_buf;
    VkDeviceMemory download_mem;
    vkenv_createBuffer(&download_buf, dev, 0, out_bytes,
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_SHARING_MODE_EXCLUSIVE, 0, NULL);
    alloc_and_bind_buffer_memory(dev, download_buf, &download_mem,
                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                 VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // 7. Descriptor set layout + pool
    VkDescriptorSetLayoutBinding bindings[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
         .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
    };
    VkDescriptorSetLayoutCreateInfo dslci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 2, .pBindings = bindings,
    };
    VkDescriptorSetLayout dsl;
    vkCreateDescriptorSetLayout(dev->device, &dslci, NULL, &dsl);

    VkDescriptorPoolSize pool_sizes[2] = {
        {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 4},
        {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 4},
    };
    VkDescriptorPoolCreateInfo dpci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
        .maxSets = 4, .poolSizeCount = 2, .pPoolSizes = pool_sizes,
    };
    VkDescriptorPool dpool;
    vkCreateDescriptorPool(dev->device, &dpci, NULL, &dpool);

    // 8. Shader + pipeline
    VkShaderModule shader_mod;
    if (!vkenv_createShaderModule(dev->device,
                                   "shaders/AffineWarp.comp.spv",
                                   &shader_mod)) {
        fprintf(stderr, "vkenv_createShaderModule failed\n"); return 1;
    }
    VkPipelineLayout playout;
    VkPipeline pipeline;
    if (!vkenv_createComputePipeline(dev->device, shader_mod, dsl,
                                      sizeof(PushConst), &playout, &pipeline)) {
        fprintf(stderr, "vkenv_createComputePipeline failed\n"); return 1;
    }

    // ===== Test case 1: pure integer translation =====
    PushConst pc1 = {OUT_W, OUT_H, 1.f, 0.f, 5.f, 0.f, 1.f, 3.f, 0.0f};
    float err1 = run_test_case(dev, cmd_pool, dpool, dsl, playout, pipeline,
                                sampler, in_view, out_view, in_image, out_image,
                                download_buf, download_mem, pc1, in_data,
                                "translate(+5,+3)");

    // ===== Test case 2: anisotropic downscale by 1.5 in each axis =====
    PushConst pc2 = {OUT_W, OUT_H, 1.5f, 0.f, 0.f, 0.f, 1.5f, 0.f, 0.25f};
    float err2 = run_test_case(dev, cmd_pool, dpool, dsl, playout, pipeline,
                                sampler, in_view, out_view, in_image, out_image,
                                download_buf, download_mem, pc2, in_data,
                                "scale(1.5x)");

    // Verdict: tolerate 1e-4 for bilinear interpolation rounding.
    bool pass = (err1 >= 0 && err1 < 1e-4f) && (err2 >= 0 && err2 < 1e-4f);
    printf("Verdict: %s\n", pass ? "PASS" : "FAIL");

    // Cleanup (best-effort)
    vkDestroyPipeline(dev->device, pipeline, NULL);
    vkDestroyPipelineLayout(dev->device, playout, NULL);
    vkDestroyShaderModule(dev->device, shader_mod, NULL);
    vkDestroyDescriptorPool(dev->device, dpool, NULL);
    vkDestroyDescriptorSetLayout(dev->device, dsl, NULL);
    vkDestroySampler(dev->device, sampler, NULL);
    vkDestroyBuffer(dev->device, upload_buf, NULL);
    vkFreeMemory(dev->device, upload_mem, NULL);
    vkDestroyBuffer(dev->device, download_buf, NULL);
    vkFreeMemory(dev->device, download_mem, NULL);
    vkDestroyImageView(dev->device, in_view, NULL);
    vkDestroyImageView(dev->device, out_view, NULL);
    vkDestroyImage(dev->device, in_image, NULL);
    vkFreeMemory(dev->device, in_mem, NULL);
    vkDestroyImage(dev->device, out_image, NULL);
    vkFreeMemory(dev->device, out_mem, NULL);
    vkDestroyCommandPool(dev->device, cmd_pool, NULL);
    free(in_data);
    vkenv_destroyDevice(&dev);
    vkenv_destroyInstance();
    return pass ? 0 : 1;
}
