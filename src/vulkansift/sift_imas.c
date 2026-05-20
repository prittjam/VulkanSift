#include "sift_imas.h"

#include "sift_detector.h"
#include "vkenv/logger.h"
#include "vkenv/vulkan_utils.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static const char LOG_TAG[] = "sift_imas";

// (Phase B-2): the per-shader push-constant structs that used to live here
// have been replaced by a single WarpParamsUBO bound at set = 1, binding = 0
// in every IMAS-chain shader. `vksift_runImasWarp` now builds one
// WarpParamsUBO, memcpys it into the per-slot uniform buffer, then dispatches
// the chain — all 5 shaders see the same parameter block.

// =============================================================================
// Helpers — descriptor layout + pipeline construction
// =============================================================================

static bool create_two_image_layout(VkDevice device, VkDescriptorType in_type, VkDescriptorType out_type,
                                    VkDescriptorSetLayout *layout)
{
  VkDescriptorSetLayoutBinding bindings[2] = {
      {.binding = 0, .descriptorType = in_type,  .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
      {.binding = 1, .descriptorType = out_type, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
  VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 2, .pBindings = bindings};
  return vkCreateDescriptorSetLayout(device, &info, NULL, layout) == VK_SUCCESS;
}

static bool create_one_image_layout(VkDevice device, VkDescriptorSetLayout *layout)
{
  VkDescriptorSetLayoutBinding binding = {
      .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
      .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
  VkDescriptorSetLayoutCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .bindingCount = 1, .pBindings = &binding};
  return vkCreateDescriptorSetLayout(device, &info, NULL, layout) == VK_SUCCESS;
}

static bool create_pool_with_sizes(VkDevice device, const VkDescriptorPoolSize *sizes, uint32_t nsizes,
                                   uint32_t max_sets, VkDescriptorPool *pool)
{
  VkDescriptorPoolCreateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
      .maxSets = max_sets, .poolSizeCount = nsizes, .pPoolSizes = sizes};
  return vkCreateDescriptorPool(device, &info, NULL, pool) == VK_SUCCESS;
}

static bool alloc_set(VkDevice device, VkDescriptorPool pool, VkDescriptorSetLayout layout, VkDescriptorSet *set)
{
  VkDescriptorSetAllocateInfo info = {
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = pool, .descriptorSetCount = 1, .pSetLayouts = &layout};
  return vkAllocateDescriptorSets(device, &info, set) == VK_SUCCESS;
}

// Build a compute pipeline whose layout binds two descriptor sets:
//   set = 0 : the per-pass image bindings (sampler/storage)
//   set = 1 : the shared WarpParamsUBO (per-slot)
// Phase B-2 dropped per-shader push-constant ranges since every parameter
// lives in the UBO now.
static bool make_pipeline(VkDevice device, const char *shader_path,
                          VkDescriptorSetLayout set0_layout,
                          VkDescriptorSetLayout set1_layout,
                          VkPipelineLayout *pl_layout, VkPipeline *pipeline)
{
  VkShaderModule shader_module;
  if (!vkenv_createShaderModule(device, shader_path, &shader_module))
  {
    logError(LOG_TAG, "Failed to create shader module: %s", shader_path);
    return false;
  }
  bool ok = vkenv_createComputePipeline2(device, shader_module, set0_layout, set1_layout,
                                         0u, pl_layout, pipeline);
  vkDestroyShaderModule(device, shader_module, NULL);
  return ok;
}

static void write_sampler_storage(VkDevice device, VkDescriptorSet set, VkSampler sampler,
                                  VkImageView in_view, VkImageView out_view)
{
  VkDescriptorImageInfo in_info  = {.sampler = sampler, .imageView = in_view,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo out_info = {.sampler = VK_NULL_HANDLE, .imageView = out_view,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1, .pImageInfo = &in_info},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .pImageInfo = &out_info}};
  vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
}

static void write_two_storage(VkDevice device, VkDescriptorSet set, VkImageView in_view, VkImageView out_view)
{
  VkDescriptorImageInfo in_info  = {.imageView = in_view,  .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkDescriptorImageInfo out_info = {.imageView = out_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet writes[2] = {
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .pImageInfo = &in_info},
      {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 1,
       .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .pImageInfo = &out_info}};
  vkUpdateDescriptorSets(device, 2, writes, 0, NULL);
}

static void write_one_storage(VkDevice device, VkDescriptorSet set, VkImageView view)
{
  VkDescriptorImageInfo info = {.imageView = view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
  VkWriteDescriptorSet w = {
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set, .dstBinding = 0,
      .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1, .pImageInfo = &info};
  vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
}

// =============================================================================
// Init / destroy
// =============================================================================

vksift_ImasPipeline vksift_createImasPipeline(vkenv_Device dev, vksift_SiftMemory mem, VkSampler sampler)
{
  vksift_ImasPipeline p = (vksift_ImasPipeline)calloc(1, sizeof(*p));
  if (!p) return NULL;
  p->dev = dev;
  p->mem = mem;
  p->sampler = sampler;

  VkDevice device = dev->device;

  // ----- Shared WarpParamsUBO descriptor set layout (set = 1 in every shader) -----
  // Single binding 0 = uniform buffer. Layout is shared across all 5 IMAS
  // pipelines; we allocate one descriptor set per pyramid slot up-front so
  // future parallel waves can each bind their own slot's UBO buffer without
  // touching descriptor state at submit time.
  {
    VkDescriptorSetLayoutBinding ubo_binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    VkDescriptorSetLayoutCreateInfo ubo_layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &ubo_binding};
    if (vkCreateDescriptorSetLayout(device, &ubo_layout_info, NULL,
                                    &p->warp_ubo_desc_set_layout) != VK_SUCCESS)
      goto fail;
  }
  {
    uint32_t n_slots = mem->nb_pyramid_slots;
    if (n_slots == 0u) n_slots = 1u;
    VkDescriptorPoolSize ubo_pool_size = {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = n_slots};
    if (!create_pool_with_sizes(device, &ubo_pool_size, 1, n_slots, &p->warp_ubo_desc_pool)) goto fail;
    for (uint32_t s = 0u; s < n_slots; ++s)
    {
      if (!alloc_set(device, p->warp_ubo_desc_pool, p->warp_ubo_desc_set_layout, &p->warp_ubo_desc_sets[s])) goto fail;
      // Bind this slot's UBO buffer into its descriptor set.
      VkDescriptorBufferInfo bi = {.buffer = mem->slots[s].warp_params_ubo,
                                   .offset = 0, .range = VK_WHOLE_SIZE};
      VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                .dstSet = p->warp_ubo_desc_sets[s],
                                .dstBinding = 0, .descriptorCount = 1,
                                .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                                .pBufferInfo = &bi};
      vkUpdateDescriptorSets(device, 1, &w, 0, NULL);
    }
  }

  // Effective slot count for Phase C-1: allocate one descriptor set per active
  // pyramid slot, each bound to that slot's image views. Pool sizes are scaled
  // by N (not VKSIFT_MAX_PYRAMID_SLOTS — only the active slots' images exist).
  uint32_t N = mem->nb_pyramid_slots;
  if (N == 0u) N = 1u;

  // ----- AffineWarp layout (sampler + storage), pool, set, pipeline -----
  if (!create_two_image_layout(device, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               &p->warp_layout)) goto fail;
  {
    VkDescriptorPoolSize sizes[2] = {{.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = N},
                                     {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = N}};
    if (!create_pool_with_sizes(device, sizes, 2, N, &p->warp_pool)) goto fail;
  }
  for (uint32_t s = 0u; s < N; ++s)
  {
    if (!alloc_set(device, p->warp_pool, p->warp_layout, &p->warp_set[s])) goto fail;
  }
  if (!make_pipeline(device, "shaders/AffineWarp.comp.spv", p->warp_layout, p->warp_ubo_desc_set_layout,
                     &p->warp_pipeline_layout, &p->warp_pipeline)) goto fail;
  // IMAS samples cached_input_image (kept in sync by recCopyInputImageCmds)
  // instead of input_image, so the on-IMAS detect path can overwrite
  // input_image with quantized tilted content without corrupting the next
  // warp's IMAS source. The cache is updated whenever a regular
  // vksift_detectFeatures uploads new content. Each slot's set writes the
  // rotated image of THAT slot, so concurrent waves don't collide.
  for (uint32_t s = 0u; s < N; ++s)
  {
    write_sampler_storage(device, p->warp_set[s], sampler, mem->cached_input_image_view, mem->slots[s].rotated_image_view);
  }

  // ----- GaussBlur1DStorage — two storage images (rotated → tilted) -----
  if (!create_two_image_layout(device, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               &p->blur_layout)) goto fail;
  {
    VkDescriptorPoolSize sizes[1] = {{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 2u * N}};
    if (!create_pool_with_sizes(device, sizes, 1, N, &p->blur_pool)) goto fail;
  }
  for (uint32_t s = 0u; s < N; ++s)
  {
    if (!alloc_set(device, p->blur_pool, p->blur_layout, &p->blur_set[s])) goto fail;
  }
  if (!make_pipeline(device, "shaders/GaussBlur1DStorage.comp.spv", p->blur_layout, p->warp_ubo_desc_set_layout,
                     &p->blur_pipeline_layout, &p->blur_pipeline)) goto fail;
  for (uint32_t s = 0u; s < N; ++s)
  {
    write_two_storage(device, p->blur_set[s], mem->slots[s].rotated_image_view, mem->slots[s].tilted_image_view);
  }

  // ----- FinvsplineRow / FinvsplineCol — single storage image, in-place IIR -----
  if (!create_one_image_layout(device, &p->finvspline_layout)) goto fail;
  {
    VkDescriptorPoolSize sizes[1] = {{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 2u * N}};
    if (!create_pool_with_sizes(device, sizes, 1, 2u * N, &p->finvspline_pool)) goto fail;
  }
  for (uint32_t s = 0u; s < N; ++s)
  {
    if (!alloc_set(device, p->finvspline_pool, p->finvspline_layout, &p->finvspline_row_set[s])) goto fail;
    if (!alloc_set(device, p->finvspline_pool, p->finvspline_layout, &p->finvspline_col_set[s])) goto fail;
  }
  if (!make_pipeline(device, "shaders/FinvsplineRow.comp.spv", p->finvspline_layout, p->warp_ubo_desc_set_layout,
                     &p->finvspline_pipeline_layout, &p->finvspline_row_pipeline)) goto fail;
  // Column pipeline reuses the row's pipeline_layout (identical push consts + layout).
  {
    VkShaderModule col_module;
    if (!vkenv_createShaderModule(device, "shaders/FinvsplineCol.comp.spv", &col_module)) goto fail;
    VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = p->finvspline_pipeline_layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = col_module, .pName = "main"}};
    VkResult res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &p->finvspline_col_pipeline);
    vkDestroyShaderModule(device, col_module, NULL);
    if (res != VK_SUCCESS) goto fail;
  }
  for (uint32_t s = 0u; s < N; ++s)
  {
    write_one_storage(device, p->finvspline_row_set[s], mem->slots[s].tilted_image_view);
    write_one_storage(device, p->finvspline_col_set[s], mem->slots[s].tilted_image_view);
  }

  // ----- FprojCubicY — sampler in, storage out, tilted_image → rotated_image -----
  // Shader binds: binding 0 = storage img_coeffs, binding 1 = storage img_out (per FprojCubicY.comp).
  if (!create_two_image_layout(device, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                               &p->fproj_layout)) goto fail;
  {
    VkDescriptorPoolSize sizes[1] = {{.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 2u * N}};
    if (!create_pool_with_sizes(device, sizes, 1, N, &p->fproj_pool)) goto fail;
  }
  for (uint32_t s = 0u; s < N; ++s)
  {
    if (!alloc_set(device, p->fproj_pool, p->fproj_layout, &p->fproj_set[s])) goto fail;
  }
  if (!make_pipeline(device, "shaders/FprojCubicY.comp.spv", p->fproj_layout, p->warp_ubo_desc_set_layout,
                     &p->fproj_pipeline_layout, &p->fproj_pipeline)) goto fail;
  for (uint32_t s = 0u; s < N; ++s)
  {
    write_two_storage(device, p->fproj_set[s], mem->slots[s].tilted_image_view, mem->slots[s].rotated_image_view);
  }

  // (The same set-0 writes are encapsulated in vksift_imasRefreshDescriptorSets
  // below for callers to rewire after a pyramid resize destroys + recreates the
  // image views these descriptors reference.)

  // Bilinear-fproj alternate pipeline (same layout + descriptor set).
  // FprojBilinearY.comp reads input as storage image (binding 0, r32f) and
  // writes output as storage image (binding 1, r32f), matching the cubic
  // shader's binding signature.
  {
    VkShaderModule bl_module;
    if (!vkenv_createShaderModule(device, "shaders/FprojBilinearY.comp.spv", &bl_module)) goto fail;
    VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = p->fproj_pipeline_layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = bl_module, .pName = "main"}};
    VkResult br = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, NULL, &p->fproj_bilinear_pipeline);
    vkDestroyShaderModule(device, bl_module, NULL);
    if (br != VK_SUCCESS) goto fail;
  }

  // ----- Readback buffer (host-visible, persistent map) -----
  p->readback_size = (VkDeviceSize)mem->slots[0].rotated_image_max_width * mem->slots[0].rotated_image_max_height * sizeof(float);
  if (!vkenv_createBuffer(&p->readback_buffer, dev, 0, p->readback_size,
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_SHARING_MODE_EXCLUSIVE, 0, NULL))
    goto fail;
  {
    VkMemoryRequirements mreq;
    vkGetBufferMemoryRequirements(device, p->readback_buffer, &mreq);
    uint32_t mtype = 0;
    if (!vkenv_findValidMemoryType(dev->physical_device, mreq,
                                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &mtype))
      goto fail;
    if (!vkenv_allocateMemory(&p->readback_memory, dev, mreq.size, mtype)) goto fail;
    if (!vkenv_bindBufferMemory(dev, p->readback_buffer, p->readback_memory, 0)) goto fail;
    if (vkMapMemory(device, p->readback_memory, 0, p->readback_size, 0, &p->readback_ptr) != VK_SUCCESS) goto fail;
  }

  // ----- Command pool + buffer + fence -----
  {
    VkCommandPoolCreateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = dev->general_queues_family_idx,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    if (vkCreateCommandPool(device, &info, NULL, &p->cmd_pool) != VK_SUCCESS) goto fail;
  }
  {
    VkCommandBufferAllocateInfo info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = p->cmd_pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(device, &info, &p->cmd_buf) != VK_SUCCESS) goto fail;
  }
  {
    VkFenceCreateInfo info = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(device, &info, NULL, &p->done_fence) != VK_SUCCESS) goto fail;
  }

  p->created = true;
  logInfo(LOG_TAG, "IMAS pipeline initialised (rotated buf %ux%u, readback %llu MB)",
          mem->slots[0].rotated_image_max_width, mem->slots[0].rotated_image_max_height,
          (unsigned long long)(p->readback_size / (1024 * 1024)));
  return p;

fail:
  vksift_destroyImasPipeline(&p);
  return NULL;
}

void vksift_imasRefreshDescriptorSets(vksift_ImasPipeline p)
{
  if (!p || !p->dev) return;
  VkDevice device = p->dev->device;
  vksift_SiftMemory mem = p->mem;
  if (!mem || mem->cached_input_image_view == VK_NULL_HANDLE) return;
  // Mirrors the writes done in vksift_createImasPipeline. The IMAS shaders'
  // set-0 image views go stale whenever vksift_prepareSiftMemoryForDetection
  // resizes the per-slot images on a canvas change; this routine rewires the
  // descriptor sets to the freshly-created views before the next dispatch.
  // Phase C-1: loop over all active slots and rewire each set against its own
  // slot's image views.
  uint32_t N = mem->nb_pyramid_slots;
  if (N == 0u) N = 1u;
  for (uint32_t s = 0u; s < N; ++s)
  {
    write_sampler_storage(device, p->warp_set[s], p->sampler,
                          mem->cached_input_image_view, mem->slots[s].rotated_image_view);
    write_two_storage(device, p->blur_set[s],
                      mem->slots[s].rotated_image_view, mem->slots[s].tilted_image_view);
    write_one_storage(device, p->finvspline_row_set[s], mem->slots[s].tilted_image_view);
    write_one_storage(device, p->finvspline_col_set[s], mem->slots[s].tilted_image_view);
    write_two_storage(device, p->fproj_set[s],
                      mem->slots[s].tilted_image_view, mem->slots[s].rotated_image_view);
  }
}

void vksift_destroyImasPipeline(vksift_ImasPipeline *pipeline_ptr)
{
  if (!pipeline_ptr || !*pipeline_ptr) return;
  vksift_ImasPipeline p = *pipeline_ptr;
  VkDevice device = p->dev ? p->dev->device : VK_NULL_HANDLE;
  if (device != VK_NULL_HANDLE)
  {
    if (p->done_fence)  vkDestroyFence(device, p->done_fence, NULL);
    if (p->cmd_pool)    vkDestroyCommandPool(device, p->cmd_pool, NULL);
    if (p->readback_memory)
    {
      if (p->readback_ptr) vkUnmapMemory(device, p->readback_memory);
      vkFreeMemory(device, p->readback_memory, NULL);
    }
    if (p->readback_buffer) vkDestroyBuffer(device, p->readback_buffer, NULL);

    if (p->fproj_bilinear_pipeline) vkDestroyPipeline(device, p->fproj_bilinear_pipeline, NULL);
    if (p->fproj_pipeline)        vkDestroyPipeline(device, p->fproj_pipeline, NULL);
    if (p->fproj_pipeline_layout) vkDestroyPipelineLayout(device, p->fproj_pipeline_layout, NULL);
    if (p->fproj_pool)            vkDestroyDescriptorPool(device, p->fproj_pool, NULL);
    if (p->fproj_layout)          vkDestroyDescriptorSetLayout(device, p->fproj_layout, NULL);

    if (p->finvspline_col_pipeline) vkDestroyPipeline(device, p->finvspline_col_pipeline, NULL);
    if (p->finvspline_row_pipeline) vkDestroyPipeline(device, p->finvspline_row_pipeline, NULL);
    if (p->finvspline_pipeline_layout) vkDestroyPipelineLayout(device, p->finvspline_pipeline_layout, NULL);
    if (p->finvspline_pool)       vkDestroyDescriptorPool(device, p->finvspline_pool, NULL);
    if (p->finvspline_layout)     vkDestroyDescriptorSetLayout(device, p->finvspline_layout, NULL);

    if (p->blur_pipeline)         vkDestroyPipeline(device, p->blur_pipeline, NULL);
    if (p->blur_pipeline_layout)  vkDestroyPipelineLayout(device, p->blur_pipeline_layout, NULL);
    if (p->blur_pool)             vkDestroyDescriptorPool(device, p->blur_pool, NULL);
    if (p->blur_layout)           vkDestroyDescriptorSetLayout(device, p->blur_layout, NULL);

    if (p->warp_pipeline)         vkDestroyPipeline(device, p->warp_pipeline, NULL);
    if (p->warp_pipeline_layout)  vkDestroyPipelineLayout(device, p->warp_pipeline_layout, NULL);
    if (p->warp_pool)             vkDestroyDescriptorPool(device, p->warp_pool, NULL);
    if (p->warp_layout)           vkDestroyDescriptorSetLayout(device, p->warp_layout, NULL);

    if (p->warp_ubo_desc_pool)         vkDestroyDescriptorPool(device, p->warp_ubo_desc_pool, NULL);
    if (p->warp_ubo_desc_set_layout)   vkDestroyDescriptorSetLayout(device, p->warp_ubo_desc_set_layout, NULL);
  }
  free(p);
  *pipeline_ptr = NULL;
}

// =============================================================================
// Rotation canvas geometry (mirror of frot.cpp's bound() + imas_cpu.jl's frot)
// =============================================================================

// Compute extended rotated canvas extents (W_rot × H_rot) for the given input
// dims and rotation angle. Output (xmin, xmax, ymin, ymax) encode the canvas
// extent in input-frame coordinates; sx, sy are the canvas dimensions.
// Exposed via sift_imas.h (Phase B-3) as vksift_imasComputeRotatedCanvas for
// the fused-cmd-buffer driver in sift_detector.c.
void vksift_imasComputeRotatedCanvas(uint32_t nx, uint32_t ny, float ca, float sa,
                                     int *xmin, int *xmax, int *ymin, int *ymax,
                                     uint32_t *sx, uint32_t *sy)
{
  int xn = 0, xm = 0, yn = 0, ym = 0;
  const int corners_x[3] = {(int)nx - 1, 0, (int)nx - 1};
  const int corners_y[3] = {0, (int)ny - 1, (int)ny - 1};
  for (int i = 0; i < 3; ++i)
  {
    int rx = (int)floorf(ca * (float)corners_x[i] + sa * (float)corners_y[i]);
    int ry = (int)floorf(-sa * (float)corners_x[i] + ca * (float)corners_y[i]);
    if (rx < xn) xn = rx;
    if (rx > xm) xm = rx;
    if (ry < yn) yn = ry;
    if (ry > ym) ym = ry;
  }
  *xmin = xn; *xmax = xm; *ymin = yn; *ymax = ym;
  *sx = (uint32_t)(xm - xn + 1);
  *sy = (uint32_t)(ym - yn + 1);
}

// =============================================================================
// Dispatch — record + submit the 5-shader IMAS chain
// =============================================================================

static void barrier_storage_to_sampled(VkCommandBuffer cmd, VkImage image)
{
  VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(image,
      VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, NULL, 0, NULL, 1, &b);
}

static void barrier_to_general_undef(VkCommandBuffer cmd, VkImage image)
{
  VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(image,
      0, VK_ACCESS_SHADER_WRITE_BIT,
      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
      (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       0, 0, NULL, 0, NULL, 1, &b);
}

// Debug stage gate. Set the env var VKSIFT_IMAS_STAGE=1|2|3|4|5 to stop after
// that stage and read back the intermediate. Unset/0 = full pipeline.
static int imas_stage_limit(void)
{
  const char *s = getenv("VKSIFT_IMAS_STAGE");
  return s ? atoi(s) : 0;
}

// Bilinear-fproj selector. Set VKSIFT_IMAS_BILINEAR=1 to skip the cubic
// IIR (FinvsplineRow + FinvsplineCol) and use a parallel bilinear resample
// in step 5 instead of FprojCubicY. Saves the serial IIR cost.
static bool imas_bilinear_mode(void)
{
  const char *s = getenv("VKSIFT_IMAS_BILINEAR");
  return s && atoi(s) != 0;
}

bool vksift_runImasWarp(vksift_ImasPipeline p, uint32_t W, uint32_t H,
                        float t_factor, float theta_rad, uint32_t *out_w, uint32_t *out_h)
{
  int  stage_limit  = imas_stage_limit();
  bool use_bilinear = imas_bilinear_mode();
  if (!p || !p->created) return false;

  float ca = cosf(theta_rad);
  float sa = sinf(theta_rad);

  // Rotation canvas (extended).
  int xmin, xmax, ymin, ymax;
  uint32_t W_rot, H_rot;
  vksift_imasComputeRotatedCanvas(W, H, ca, sa, &xmin, &xmax, &ymin, &ymax, &W_rot, &H_rot);

  // Output (tilted) dims.
  uint32_t H_t = (t_factor > 1.0f) ? (uint32_t)floorf((float)H_rot / t_factor) : H_rot;
  if (H_t < 1) H_t = 1;
  *out_w = W_rot;
  *out_h = H_t;

  // Inverse rotation: for output pixel (x_out, y_out) ∈ [0, W_rot) × [0, H_rot),
  // the original input coord is (xp, yp) = R(θ) · (x_out + xmin, y_out + ymin).
  // Matches frot.cpp lines 68-77: xp = ca·x − sa·y, yp = sa·x + ca·y, with
  // x = (x_out + xmin), y = (y_out + ymin), since xtrans = ytrans = 0 in the
  // expanded-canvas branch.
  float a11 = ca;
  float a12 = -sa;
  float a13 = ca * (float)xmin - sa * (float)ymin;
  float a21 = sa;
  float a22 = ca;
  float a23 = sa * (float)xmin + ca * (float)ymin;
  // fast_imas's frot uses bg = *b for OOB (caller passes 0.5 or 128 / 255 in
  // imas_cpu.jl). We mirror imas_cpu.jl's FROT_FILL = 0.5.
  float fill = 0.5f;

  // σ_aa = 0.8·√(t²-1) for t > 1, else 0 (identity → skip blur).
  float sigma_aa = (t_factor > 1.0f) ? 0.8f * sqrtf(t_factor * t_factor - 1.0f) : 0.0f;

  // ===== Populate the per-slot WarpParamsUBO (slot 0 in Phase B-2) =====
  // A single UBO write covers all 5 IMAS-chain shaders for this warp because
  // every shader reads the same canvas / matrix / sigma fields. HOST_COHERENT
  // memory was used at allocation time so the write is immediately visible
  // to the GPU when the cmd buffer executes (no flush required).
  {
    WarpParamsUBO ubo_data = {0};
    ubo_data.a11 = a11; ubo_data.a12 = a12; ubo_data.a13 = a13;
    ubo_data.a21 = a21; ubo_data.a22 = a22; ubo_data.a23 = a23;
    ubo_data.fill_value    = fill;       // AffineWarp OOB fill (= 0.5)
    ubo_data.fproj_bg_value = 0.0f;      // Fproj OOB fill — preserves Phase A behaviour
    ubo_data.W_rot         = W_rot;
    ubo_data.H_rot         = H_rot;
    ubo_data.H_sub         = H_t;
    // Quantize fields are unused by sift_imas (the detector's on-IMAS path
    // owns Quantize), but populate them anyway so the slot's UBO is in a
    // consistent state if any consumer reads it.
    ubo_data.canvas_w      = p->mem->curr_input_image_width;
    ubo_data.canvas_h      = p->mem->curr_input_image_height;
    ubo_data.valid_w       = W_rot;
    ubo_data.valid_h       = H_t;
    ubo_data.warp_idx      = 0u;
    ubo_data.sigma_aa      = sigma_aa;
    ubo_data.t_factor      = t_factor;
    ubo_data.quantize_fill = 0.5f;
    ubo_data.gauss_dir_x   = 0.0f;
    ubo_data.gauss_dir_y   = 1.0f;  // IMAS vertical σ_aa blur
    memcpy(p->mem->slots[0].warp_params_ubo_ptr, &ubo_data, sizeof(WarpParamsUBO));
  }

  // ===== Record command buffer =====
  if (vkResetCommandBuffer(p->cmd_buf, 0) != VK_SUCCESS) return false;
  {
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (vkBeginCommandBuffer(p->cmd_buf, &bi) != VK_SUCCESS) return false;
  }

  // Layout transitions:
  // - input_image: must be GENERAL for sampler reads. After vksift_detectFeatures
  //   finishes, the previous pipeline leaves it in SHADER_READ_ONLY_OPTIMAL or
  //   GENERAL depending on which stage was last. We force GENERAL with a
  //   barrier from UNDEFINED → GENERAL, which preserves contents (Vulkan
  //   spec says contents preserved if old_layout == GENERAL, undefined
  //   otherwise — but vkCmdPipelineBarrier with src access = 0 doesn't
  //   actually invalidate data; only the layout transition matters here).
  //   To be safe, we transition from whatever (SHADER_READ_ONLY_OPTIMAL is
  //   commonly used post-detect) → GENERAL using GENERAL as old_layout
  //   (no-op transition that just emits memory-barrier semantics).
  {
    VkImageMemoryBarrier b = vkenv_genImageMemoryBarrier(p->mem->slots[0].input_image,
        VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
        VK_ACCESS_SHADER_READ_BIT,
        VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        (VkImageSubresourceRange){VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1});
    vkCmdPipelineBarrier(p->cmd_buf,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, NULL, 0, NULL, 1, &b);
  }
  // rotated/tilted scratch images: ensure GENERAL layout. We don't preserve
  // contents (overwritten in step 1 / step 2 fully).
  barrier_to_general_undef(p->cmd_buf, p->mem->slots[0].rotated_image);
  barrier_to_general_undef(p->cmd_buf, p->mem->slots[0].tilted_image);

  // ----- 1. AffineWarp (rotate) : input_image → rotated_image -----
  // Legacy single-slot path: bind slot[0]'s descriptor sets (vksift_runImasWarp
  // is only entered via the Phase A non-fused IMAS readback path and the JL FFI
  // vksift_jl_run_imas, neither of which expose slot selection).
  {
    vkCmdBindPipeline(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->warp_pipeline);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->warp_pipeline_layout,
                            0, 1, &p->warp_set[0], 0, NULL);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->warp_pipeline_layout,
                            1, 1, &p->warp_ubo_desc_sets[0], 0, NULL);
    vkCmdDispatch(p->cmd_buf, (W_rot + 7) / 8, (H_rot + 7) / 8, 1);
  }
  barrier_storage_to_sampled(p->cmd_buf, p->mem->slots[0].rotated_image);

  if (stage_limit == 1) {
    // Read back rotated_image directly
    VkBufferImageCopy region = {.bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {W_rot, H_rot, 1}};
    vkCmdCopyImageToBuffer(p->cmd_buf, p->mem->slots[0].rotated_image, VK_IMAGE_LAYOUT_GENERAL,
                           p->readback_buffer, 1, &region);
    *out_h = H_rot;
    goto submit;
  }

  // ----- 2. GaussBlur1DStorage vertical : rotated_image → tilted_image -----
  // At σ_aa = 0 (identity tilt) shader degenerates to a copy via fetch_clamped.
  {
    vkCmdBindPipeline(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->blur_pipeline);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->blur_pipeline_layout,
                            0, 1, &p->blur_set[0], 0, NULL);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->blur_pipeline_layout,
                            1, 1, &p->warp_ubo_desc_sets[0], 0, NULL);
    vkCmdDispatch(p->cmd_buf, (W_rot + 7) / 8, (H_rot + 7) / 8, 1);
  }
  barrier_storage_to_sampled(p->cmd_buf, p->mem->slots[0].tilted_image);

  if (stage_limit == 2) {
    VkBufferImageCopy region = {.bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {W_rot, H_rot, 1}};
    vkCmdCopyImageToBuffer(p->cmd_buf, p->mem->slots[0].tilted_image, VK_IMAGE_LAYOUT_GENERAL,
                           p->readback_buffer, 1, &region);
    *out_h = H_rot;
    goto submit;
  }

  if (!use_bilinear)
  {
    // ----- 3. FinvsplineRow : tilted_image in-place per-row IIR -----
    {
      vkCmdBindPipeline(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_row_pipeline);
      vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_pipeline_layout,
                              0, 1, &p->finvspline_row_set[0], 0, NULL);
      vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_pipeline_layout,
                              1, 1, &p->warp_ubo_desc_sets[0], 0, NULL);
      vkCmdDispatch(p->cmd_buf, (H_rot + 63) / 64, 1, 1);
    }
    barrier_storage_to_sampled(p->cmd_buf, p->mem->slots[0].tilted_image);

    if (stage_limit == 3) {
      VkBufferImageCopy region = {.bufferOffset = 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                  .imageExtent = {W_rot, H_rot, 1}};
      vkCmdCopyImageToBuffer(p->cmd_buf, p->mem->slots[0].tilted_image, VK_IMAGE_LAYOUT_GENERAL,
                             p->readback_buffer, 1, &region);
      *out_h = H_rot;
      goto submit;
    }

    // ----- 4. FinvsplineCol : tilted_image in-place per-col IIR -----
    {
      vkCmdBindPipeline(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_col_pipeline);
      vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_pipeline_layout,
                              0, 1, &p->finvspline_col_set[0], 0, NULL);
      vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->finvspline_pipeline_layout,
                              1, 1, &p->warp_ubo_desc_sets[0], 0, NULL);
      vkCmdDispatch(p->cmd_buf, (W_rot + 63) / 64, 1, 1);
    }
    barrier_storage_to_sampled(p->cmd_buf, p->mem->slots[0].tilted_image);
  }

  // ----- 5. Fproj{Cubic|Bilinear}Y : tilted_image → rotated_image -----
  // NOTE: FprojCubicY / FprojBilinearY use ubo.fill_value (= 0.5 — same as
  // AffineWarp) for OOB samples now, instead of the previous bg_value = 0.
  // We compensated for this in the IMAS-25 covering by ensuring fill is the
  // background mid-gray; if 0.0 was specifically needed downstream, write
  // a separate UBO field for it.
  {
    VkPipeline fproj_pipe = use_bilinear ? p->fproj_bilinear_pipeline : p->fproj_pipeline;
    vkCmdBindPipeline(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, fproj_pipe);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->fproj_pipeline_layout,
                            0, 1, &p->fproj_set[0], 0, NULL);
    vkCmdBindDescriptorSets(p->cmd_buf, VK_PIPELINE_BIND_POINT_COMPUTE, p->fproj_pipeline_layout,
                            1, 1, &p->warp_ubo_desc_sets[0], 0, NULL);
    vkCmdDispatch(p->cmd_buf, (W_rot + 7) / 8, (H_t + 7) / 8, 1);
  }
  barrier_storage_to_sampled(p->cmd_buf, p->mem->slots[0].rotated_image);

  // ----- 6. Copy rotated_image (final tilted) → host-mapped readback buffer -----
  {
    VkBufferImageCopy region = {
        .bufferOffset = 0, .bufferRowLength = 0, .bufferImageHeight = 0,
        .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {0, 0, 0},
        .imageExtent = {W_rot, H_t, 1}};
    vkCmdCopyImageToBuffer(p->cmd_buf, p->mem->slots[0].rotated_image, VK_IMAGE_LAYOUT_GENERAL,
                           p->readback_buffer, 1, &region);
  }

submit:
  if (vkEndCommandBuffer(p->cmd_buf) != VK_SUCCESS) return false;

  // ===== Submit + wait =====
  vkResetFences(p->dev->device, 1, &p->done_fence);
  {
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                       .commandBufferCount = 1, .pCommandBuffers = &p->cmd_buf};
    if (vkQueueSubmit(p->dev->general_queues[0], 1, &si, p->done_fence) != VK_SUCCESS) return false;
  }
  if (vkWaitForFences(p->dev->device, 1, &p->done_fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
    return false;

  return true;
}
