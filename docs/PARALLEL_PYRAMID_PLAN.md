# Parallel Pyramid IMAS+Detect Pipeline — Engineering Plan

A complete, self-contained plan for landing parallel-pyramid IMAS+detect
processing in this VulkanSift fork. A fresh agent should be able to execute
this end-to-end against the `asift-batch` branch.

---

## 0. Quick orientation

- **Repo / branch**: `/home/jbpritts/src/VulkanSift`, branch `asift-batch`.
- **Last commit on `asift-batch` before this work**: `fabea2b` ("parallel-
  pyramid roadmap: declare nb_pyramid_slots in vksift_Config"). The Config
  field already exists; nothing in code consumes it yet.
- **Build**:
  ```bash
  cd /home/jbpritts/src/VulkanSift/build && make -j
  # then rebuild the JL FFI .so (CMake doesn't own it):
  cd /home/jbpritts/src/VulkanSift && \
    gcc -shared -fPIC -O2 -I include \
        -o build/libvksift_jl.so vksift_jl.c \
        -L build -lvulkansift -Wl,-rpath,'$ORIGIN'
  ```
- **Symbol check after build**:
  ```bash
  nm -D build/libvksift_jl.so | grep vksift_jl_
  ```
- **Test image**: `dataset_sonyfx3_tamron20mm_f8/jpeg/DSC02847.JPG` under
  `BlobBoards.data_dir()` (= `/home/jbpritts/informatik/Datasets/20251028_BlobBoards_Office`,
  needs `/mnt/informatik` CIFS share mounted; `sudo mount /mnt/informatik`
  if cold).
- **Julia REPL**: kaimon gate inside `/home/jbpritts/src/BlobBoards.jl`. After
  rebuilding the `.so`, **restart the gate** via `mcp__kaimon__manage_repl
  command=restart` so the new symbol is picked up.

---

## 1. Goal

For 25-warp IMAS-25 detection on the 4240×2832 test image, drop the
wallclock from the current 0.98 s to **~200 ms** while preserving CPU-IMAS
quality (recall ≥ 84%, σ ratio median ≈ 1.0, median position error ≤ 1.9
px). Achieved by:

1. **Fusing** IMAS warp + Quantize + SIFT detect into a single recorded
   command buffer per slot (no CPU↔GPU sync mid-chain).
2. **Multi-instancing** every per-pyramid GPU resource so N IMAS+detect
   chains can be in-flight at once.
3. **Indirect-dispatch + UBO-driven** parameterization so each pre-recorded
   command buffer is reusable across all 25 warps — host updates the
   parameter UBO before each submission.
4. **GPU back-projection** for the post-detect tilted→input coordinate
   transform (currently 273 ms of the 980 ms budget runs on host CPU).

---

## 2. Starting state

The single-slot path is correct and fast at 0.98 s. Recent work landed:

- `30cfa49` — boundary-aware 3D NMS in `ExtractKeypoints.comp`, σ formula
  fix, upsample-bias compensation. Quality fixes.
- `25d79d0` — `vksift_detectFeaturesOnImas` + on-device `QuantizeF32ToInput`
  shader. Eliminates the host Float32→UInt8 roundtrip.
- `dc846af` — `mem->cached_input_image` synced by `recCopyInputImageCmds`;
  IMAS pipeline reads from cached_input so detect_on_imas can overwrite
  `input_image` without corrupting the next warp's IMAS source.
- `fabea2b` — declares `vksift_Config.nb_pyramid_slots` field (default 1).

Per-stage timing for 25-warp IMAS on the test image, post-`dc846af`:

```
1. GPU IMAS warp pipeline (run_imas)                   175 ms  18%
2. Device quantize + GPU SIFT detect (on-IMAS path)    500 ms  51%
3. Back-projection + boundary filter (host CPU)        273 ms  28%
   Misc Julia overhead                                  35 ms   3%
Total                                                  983 ms
```

Target end-state per-stage (5-slot parallel + GPU back-projection):

```
1. GPU IMAS warp pipeline   ─┐                          50 ms  25%   (overlapped)
2. Device quantize + detect  ┘                         150 ms  75%   (5× parallel)
3. GPU back-projection                                  10 ms   5%
   Host (feature download + Julia)                       5 ms   2%
   (Plus ~10 ms wave dispatch overhead)
Total                                                  ~200 ms
```

---

## 3. Architecture

### 3.1 Per-slot pyramid memory

Every per-pyramid GPU resource becomes a member of a `vksift_SiftPyramidSlot`
struct, and `vksift_SiftMemory` holds an array of `nb_pyramid_slots` slots:

```c
typedef struct {
  // IMAS pipeline scratch (max W_rot × H_rot, R32F)
  VkImage         rotated_image;
  VkImageView     rotated_image_view;
  VkDeviceMemory  rotated_image_memory;
  VkDeviceSize    rotated_image_memory_size;
  uint32_t        rotated_image_max_width;
  uint32_t        rotated_image_max_height;
  VkImage         tilted_image;       // … same metadata pattern
  // …

  // Input + pre-blur + warped (all sized at curr_input_image_*)
  VkImage         input_image;        // R8_UNORM
  VkImageView     input_image_view;
  VkDeviceMemory  input_image_memory;
  VkDeviceSize    input_image_memory_size;
  VkImage         blurred_input_image;     // R32F
  VkImageView     blurred_input_image_view;
  // … plus warped_input_image and friends

  // Scale-space pyramid (per-octave arrays)
  VkImage         *blur_tmp_image_arr;
  VkImageView     *blur_tmp_image_view_arr;
  VkDeviceMemory  *blur_tmp_image_memory_arr;
  VkImage         *octave_image_arr;
  VkImageView     *octave_image_view_arr;
  VkDeviceMemory  *octave_image_memory_arr;
  VkImage         *octave_DoG_image_arr;
  // …

  // SIFT feature output (per-slot independent buffer)
  VkBuffer        sift_buffer;
  VkDeviceMemory  sift_buffer_memory;
  vksift_SiftBufferInfo sift_buffer_info;

  // Per-slot uniform buffer holding WarpParamsUBO (see §3.3)
  VkBuffer        warp_params_ubo;
  VkDeviceMemory  warp_params_ubo_memory;
  void           *warp_params_ubo_ptr;     // host-mapped, persistently coherent

  // Per-slot dispatch buffer (vkCmdDispatchIndirect targets)
  VkBuffer        dispatch_buffer;
  VkDeviceMemory  dispatch_buffer_memory;
  void           *dispatch_buffer_ptr;     // host-mapped
} vksift_SiftPyramidSlot;
```

Resources that stay **single instance** (shared across slots):

- `cached_input_image` — read-only IMAS source; only the regular detect
  upload path writes it.
- `image_staging_buffer` — host-mapped staging for the host upload (only
  hit on the regular detect path, not on the IMAS waves).
- Pipeline objects (`*_pipeline`, `*_pipeline_layout`, `*_desc_set_layout`)
  — Vulkan pipelines are immutable; one per shader, reused across slots.

Descriptor sets DO become per-slot for every pipeline that binds a per-
pyramid image view.

### 3.2 Fused IMAS+detect command buffer

One pre-recorded command buffer per slot covers:

1. IMAS chain (5 shaders): AffineWarp → GaussBlur1D(σ_aa) → FinvsplineRow
   → FinvsplineCol → FprojCubicY. Writes `slot.rotated_image`.
2. Layout transition `slot.rotated_image` → GENERAL with SHADER_READ.
3. `QuantizeF32ToInput`: writes `slot.input_image` from `slot.rotated_image`
   (replacing the existing `recCopyInputImageCmds` upload path for the
   on-IMAS variant).
4. SIFT detect pipeline (PreBlur1D → AffineWarp(identity) → scale-space
   build per octave → DoG → ExtractKeypoints → optional orientation +
   descriptors → CopySIFTCount). Writes features into `slot.sift_buffer`.
5. GPU back-projection (new): a small compute shader reads each feature
   in `slot.sift_buffer`, transforms `(f.x, f.y, σ)` from tilted-frame to
   input-frame using params from `slot.warp_params_ubo`, writes back.

All dispatch dimensions come from `slot.dispatch_buffer` via
`vkCmdDispatchIndirect`. The cmd buffer doesn't bake any warp-dependent
constants; everything dynamic is in the UBO and the dispatch buffer.

Recorded once per slot at memory init time (or when memory layout
changes), reused for every wave.

### 3.3 `WarpParamsUBO` — the central host↔GLSL contract

```c
typedef struct {
  // Affine matrix R(-φ) · diag(1,t) · R(φ) — IMAS rotate-then-tilt
  float a11, a12, a13;
  float a21, a22, a23;
  float fill_value;   // 0.5 (matches FROT_FILL convention)

  // IMAS canvas dimensions
  uint32_t W, H;          // input image dims (same for every warp in a run)
  uint32_t W_rot, H_rot;  // post-rotation canvas
  uint32_t H_sub;         // = H_rot / t after fproj subsample

  // σ_aa = 0.8 * sqrt(t² - 1); 0 at identity
  float sigma_aa;

  // Tilt factor (used by FprojCubicY)
  float t_factor;

  // Quantize push-const equivalents
  uint32_t canvas_w, canvas_h;   // = curr_input_image_width/_height
  uint32_t valid_w, valid_h;     // = (W_rot, H_sub) sub-region inside canvas
  float    quantize_fill;        // 0.5

  // Back-projection params (for the GPU compute shader)
  float cos_phi, sin_phi;
  float phi_rad;       // for synthesizing tiltedcoor2imagecoor on GPU
  // … plus anything else tiltedcoor2imagecoor needs

  // Tag for the host: the warp index this slot is processing in the
  // current wave. The back-projection shader stamps this into each
  // emitted feature's "tag" slot so the host can demux by warp_idx.
  int32_t warp_idx;

  // Padding to keep std140 alignment if needed (Vulkan UBO layout rules)
  int32_t _pad0, _pad1, _pad2;
} WarpParamsUBO;  // std140 layout, ~96 bytes
```

In GLSL each shader binds it as:
```glsl
layout(set = 1, binding = 0) uniform WarpParamsBlock {
  WarpParamsUBO p;
} ubo;
```

Pre-compute on host every wave:
- `(a11..a23)` from `(t, φ)` via the same formula in `examples/asift_gpu/driver.jl::affine_for_warp`
- `(W_rot, H_rot)` from `(W, H, φ)` via `BlobBoards._rotated_size`
- `H_sub` = `floor(H_rot / t)`
- `sigma_aa` = `t > 1 ? 0.8 * sqrt(t² - 1) : 0`
- `cos_phi`, `sin_phi`, `phi_rad`
- Dispatch group counts (next §)

### 3.4 Indirect dispatch buffer

A small SSBO per slot containing `VkDispatchIndirectCommand` triples
(x, y, z group counts) for every `vkCmdDispatch` in the fused cmd buffer:

```c
typedef struct {
  VkDispatchIndirectCommand affinewarp_dispatch;        // ceil((W_rot, H_rot)/8)
  VkDispatchIndirectCommand gaussblur_dispatch;         // ceil((W_rot, H_rot)/8)
  VkDispatchIndirectCommand finvspline_row_dispatch;    // ceil(H_rot/8), no x dim
  VkDispatchIndirectCommand finvspline_col_dispatch;    // ceil(W_rot/8), no y dim
  VkDispatchIndirectCommand fproj_dispatch;             // ceil((W_rot, H_sub)/8)
  VkDispatchIndirectCommand quantize_dispatch;          // ceil((canvas_w, canvas_h)/8)
  // SIFT detect dispatches are NOT here — they use known curr_input dims
  // baked into the recorded cmd buffer (canvas is stable across warps).
  VkDispatchIndirectCommand backproject_dispatch;       // ceil(max_features/64)
} SlotDispatchBuffer;
```

Host fills this before each wave submission. Cmd buffer references these
offsets via `vkCmdDispatchIndirect`.

### 3.5 Wave dispatch

```c
// Public API
void vksift_dispatchParallelIMAS(vksift_Instance instance,
                                 const vksift_WarpSpec *warps,
                                 uint32_t n_warps,
                                 uint32_t n_slots);
```

Loop body:
1. Host computes WarpParamsUBO + DispatchBuffer for warps in the current
   wave (slots 0..min(n_slots, remaining_warps)-1).
2. `memcpy` into each slot's host-mapped UBO + dispatch buffer.
3. `vkQueueSubmit` all the per-slot pre-recorded cmd buffers in one call
   (an array of `VkSubmitInfo`, one per slot). Single fence signals when
   all complete.
4. `vkWaitForFences`.
5. Concatenate features from all slot output buffers, tagged with their
   warp_idx (set by the GPU back-projection shader).

---

## 4. Phase A — per-slot pyramid memory

### Files touched

- `src/vulkansift/sift_memory.h` — add `vksift_SiftPyramidSlot` struct,
  change `vksift_SiftMemory` to hold `vksift_SiftPyramidSlot slots[VKSIFT_MAX_PYRAMID_SLOTS]`
  and `nb_pyramid_slots`.
- `src/vulkansift/sift_memory.c` — `setupDynamicObjectsAndMemory` becomes
  a per-slot loop. `vksift_destroyMemory` frees per slot. Helper
  `setupOneSlot(memory, slot_idx)` factors out the per-slot allocation.
- `src/vulkansift/sift_detector.{h,c}` — descriptor set arrays per slot
  for every pipeline that binds per-pyramid memory: PreBlur1D, AffineWarp,
  GaussianBlur (× max_nb_octaves), DifferenceOfGaussian (× max_nb_octaves),
  Downsample2x (× max_nb_octaves), ExtractKeypoints (× max_nb_octaves),
  ComputeOrientation, ComputeDescriptors, Quantize. Each becomes
  `[VKSIFT_MAX_PYRAMID_SLOTS][per_octave]` or `[VKSIFT_MAX_PYRAMID_SLOTS]`.
  `writeDescriptorSets` becomes `writeDescriptorSetsForSlot(slot)` and is
  called for every slot.
- `src/vulkansift/sift_imas.c` — `vksift_ImasPipeline` holds per-slot
  descriptor sets and pipeline layouts; `vksift_runImasWarp` becomes
  `vksift_runImasWarpForSlot(slot)`. (Phase A: still records its own
  cmd buffer per call; Phase B replaces with pre-recorded fused.)
- `src/vulkansift/vulkansift.c` — `vksift_detectFeatures`,
  `vksift_detectFeaturesOnImas` keep the single-slot semantics by hard-
  coding slot 0 (preserves backward compat).
- `include/vulkansift/vulkansift_types.h` — no change; field already added.

### Concrete sub-tasks

1. **Define `VKSIFT_MAX_PYRAMID_SLOTS = 8`** in `sift_memory.h` so all
   per-slot arrays can be sized statically. The runtime `nb_pyramid_slots`
   ≤ this.

2. **Move resources into the slot struct.** A non-trivial refactor since
   nearly every line of `setupDynamicObjectsAndMemory` references
   `memory->{rotated_image, input_image, blurred_input_image, ...}`. The
   mechanical change: wrap each existing allocation block in
   `for (uint32_t s = 0; s < memory->nb_pyramid_slots; s++)` and replace
   `memory->X` with `memory->slots[s].X`.

3. **Cache `cached_input_image` stays single** — pull it out of the slot
   struct (it's the shared IMAS source). Keep at `memory->cached_input_image`.

4. **Per-slot SIFT buffer**: VKS already has a `sift_buffer_arr[]` indexed
   by `gpu_buffer_id` (default 2). Reuse this — slot `s` writes to
   `sift_buffer_arr[s]`. Bump default `sift_buffer_count` to `nb_pyramid_slots`.
   `recExtractKeypointsCmds` already binds the correct buffer based on
   `detector->curr_buffer_idx`; just need to write the correct
   `curr_buffer_idx` for the slot being dispatched.

5. **Validate Phase A**: existing `vksift_detectFeatures` (single-slot
   semantics, slot 0 only) on the test image should produce the SAME
   feature set as before. Diff against the `dc846af` baseline using the
   existing Julia comparison harness:

   ```julia
   # In a fresh BlobBoards kaimon session:
   include("examples/asift_gpu/imas_cpu.jl")
   include("examples/asift_gpu/driver.jl")
   # Run the on-IMAS path with default n_pyramid_slots=1, expect
   # 9720 features after 5×5 NMS, CPU recall 84.8%.
   ```

### Acceptance criteria for Phase A

- Single-slot wallclock unchanged (~0.98 s).
- `nb_pyramid_slots = 1` produces identical features to pre-refactor.
- `nb_pyramid_slots = 2` allocates the right amount of memory without
  errors (verify with `nvidia-smi` — should see ~8 GB allocated vs ~4 GB).
- No actual concurrent dispatching yet — that's Phase C. Slot 1+ memory
  is allocated but unused.

---

## 5. Phase B — WarpParamsUBO + indirect dispatch + fused cmd buffer

### 5.1 GLSL shader migrations

Each IMAS shader replaces its push-const block with a UBO read. Side-
by-side examples:

**Before** (`AffineWarp.comp:27-37`):
```glsl
layout(push_constant) uniform PushConst {
    uint  output_width, output_height;
    float a11, a12, a13, a21, a22, a23, fill_value;
} push_const;
```

**After**:
```glsl
layout(set = 1, binding = 0, std140) uniform WarpParamsBlock {
    // … matches the C struct exactly
    mat3 affine;             // or 6 floats; std140 mat3 wastes a vec4
    float fill_value;
    uvec2 W_rot_H_rot;
    // … the full block
} ubo;
```

Shaders to migrate (in order of dependency):
- `AffineWarp.comp` (IMAS version) — affine matrix + output dims + fill
- `GaussBlur1DStorage.comp` — sigma + dir + in_w/h
- `FinvsplineRow.comp` — width, height (use W_rot, H_rot)
- `FinvsplineCol.comp` — width, height
- `FprojCubicY.comp` — t_factor, output dims, input dims, bg
- `QuantizeF32ToInput.comp` — canvas_w/h, valid_w/h, fill
- `PreBlur1D.comp` — (used by SIFT detect post-quantize; keep push_const for
  the σ_aa direction since detect's PreBlur1D is identity-σ in the on-
  IMAS path; UBO migration is optional)
- `ExtractKeypoints.comp` — keep push_const (octave_idx is static at record
  time; warp params don't reach here)

Use Vulkan's `descriptorSet=1` to keep the per-slot UBO orthogonal to the
existing `descriptorSet=0` bindings — easier to add without disturbing
existing descriptor allocations.

### 5.2 Indirect dispatch

For each `vkCmdDispatch(cmdbuf, gx, gy, gz)` whose group counts depend on
warp params, change to:

```c
vkCmdDispatchIndirect(cmdbuf, slot->dispatch_buffer,
                     offsetof(SlotDispatchBuffer, affinewarp_dispatch));
```

The per-slot `dispatch_buffer` is host-mapped; host writes the
`VkDispatchIndirectCommand` triples before submission.

Dispatches that stay direct (constant group counts): everything inside
the SIFT detect scale-space pipeline (dimensions tied to the stable
`curr_input_image_*`, not warp params). The Phase B refactor only
indirectifies the IMAS-chain + Quantize dispatches.

### 5.3 Fused IMAS+detect command buffer

A new `recImasAndDetectCmdsForSlot(detector, cmdbuf, slot_idx)` function
combines what `vksift_runImasWarp` and `recordCommandBuffers` did
separately. Pre-recorded once per slot, into a new per-slot field:

```c
// In sift_detector.h:
VkCommandBuffer fused_imas_detect_command_buffer[VKSIFT_MAX_PYRAMID_SLOTS];
```

Sequence inside the fused cmd buffer:
1. `recBufferOwnershipTransferCmds` (acquire) — only if async transfer
2. IMAS chain (now using indirect dispatch + UBO):
   - barrier `cached_input_image` → GENERAL for sampler read
   - barrier `slot.rotated_image` → GENERAL for storage write
   - AffineWarp (indirect)
   - barrier `slot.rotated_image` → SHADER_READ
   - barrier `slot.tilted_image` → GENERAL
   - GaussBlur1DStorage (indirect)
   - barrier `slot.tilted_image` → GENERAL (storage RW)
   - FinvsplineRow (indirect) — in-place on tilted
   - barrier
   - FinvsplineCol (indirect) — in-place
   - barrier `slot.tilted_image` → SHADER_READ
   - barrier `slot.rotated_image` → GENERAL (storage write — reusing as fproj output)
   - FprojCubicY (indirect) — writes to slot.rotated_image
3. Quantize chain:
   - barrier `slot.rotated_image` → GENERAL (shader read)
   - barrier `slot.input_image` → GENERAL (storage write)
   - Quantize (indirect)
   - barrier `slot.input_image` → GENERAL (shader read)
4. SIFT detect (existing recScaleSpaceConstructionCmds etc. on slot resources):
   - `recClearBufferDataCmds` for slot's sift_buffer
   - `recScaleSpaceConstructionCmds` × octaves
   - `recDifferenceOfGaussianCmds`
   - `recExtractKeypointsCmds` (writes to `sift_buffer_arr[slot_idx]`)
   - Optionally `recComputeOrientationsCmds` + `recComputeDestriptorsCmds`
5. GPU back-projection (Phase D — placeholder in Phase B):
   - For Phase B, emit nothing; the host still does back-projection.
6. `recCopySIFTCountCmds`
7. `recBufferOwnershipTransferCmds` (release) — only if async transfer

### Acceptance criteria for Phase B

- Single-slot path still produces identical features.
- A single fused cmd buffer submission for one warp matches the previous
  separate runImas + detectFeaturesOnImas pair in output.
- Wallclock for single-slot, 25 warps: roughly unchanged (no parallelism
  yet). The win is that we now have one cmd buffer per slot, ready for
  parallel submission.

---

## 6. Phase C — parallel submission API + JL FFI

### 6.1 Public C API

In `vulkansift.h`:

```c
typedef struct {
  float t_factor;
  float theta_rad;
  // pyramid input dims (canvas) — typically constant across the schedule
  // = max_W_rot × max_H_sub for the worst-case warp
} vksift_WarpSpec;

VKSIFT_EXPORT void vksift_dispatchParallelIMAS(
    vksift_Instance instance,
    const vksift_WarpSpec *warps,
    uint32_t n_warps);
```

In `vulkansift.c`:

```c
void vksift_dispatchParallelIMAS(vksift_Instance instance,
                                 const vksift_WarpSpec *warps,
                                 uint32_t n_warps) {
  uint32_t n_slots = instance->sift_memory->nb_pyramid_slots;
  for (uint32_t base = 0; base < n_warps; base += n_slots) {
    uint32_t wave = MIN(n_slots, n_warps - base);
    // 1. Update each slot's UBO + dispatch buffer for warps[base..base+wave-1]
    for (uint32_t s = 0; s < wave; s++) {
      fill_warp_params_ubo(instance, s, &warps[base + s], base + s);
      fill_dispatch_buffer  (instance, s, &warps[base + s]);
    }
    // 2. Submit all wave cmd buffers in one vkQueueSubmit (or batched submits)
    VkSubmitInfo submits[VKSIFT_MAX_PYRAMID_SLOTS];
    for (uint32_t s = 0; s < wave; s++) {
      submits[s].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submits[s].commandBufferCount = 1;
      submits[s].pCommandBuffers = &detector->fused_imas_detect_command_buffer[s];
      // … set up waits/signals, share or alternate fences
    }
    // 3. Submit, fence-wait
    vkResetFences(device, 1, &wave_fence);
    vkQueueSubmit(detector->general_queue, wave, submits, wave_fence);
    vkWaitForFences(device, 1, &wave_fence, VK_TRUE, UINT64_MAX);
    // 4. Feature buffer downloads happen in the JL layer per slot
  }
}
```

**Important**: the per-slot sift_buffer holds features for that slot's
warp. After each wave, the JL FFI reads features from each slot's
buffer using `vksift_downloadFeatures(instance, feats, slot_idx)`.

### 6.2 JL FFI

```c
// vksift_jl.h
uint32_t vksift_jl_dispatch_parallel_imas(vksift_jl_handle h,
                                          const float *t_factors,
                                          const float *theta_rads,
                                          uint32_t n_warps,
                                          /* output: */
                                          uint32_t *features_per_warp);
// Caller then calls vksift_jl_get_features_for_warp(h, buf, warp_idx)
// to download features tagged with that warp_idx.
```

### 6.3 Julia integration

Replace the per-warp loop in `examples/asift_gpu/driver.jl::asift_gpu_detect`
(or a new sibling function `asift_gpu_detect_parallel`) with a single
call to `vksift_jl_dispatch_parallel_imas` followed by per-warp feature
downloads.

### Acceptance criteria for Phase C

- 25-warp IMAS-25 wallclock drops from 0.98 s to ~300 ms with
  `n_pyramid_slots = 4`.
- Feature set matches the single-slot path within float-rounding noise
  (< 0.5% diff in total count).
- `nvidia-smi` confirms `n_pyramid_slots` instances of pyramid memory
  allocated (~3.5 GB each).

---

## 7. Phase D — GPU back-projection

### 7.1 The shader

New `BackProjectFeatures.comp`. Reads each feature in the slot's
sift_buffer, applies the inverse-rotation+translation+tilt transform
(matches `ImasCpu.tiltedcoor2imagecoor` exactly), writes:

- updated `(x, y)` in the input frame (replacing tilted-frame coords)
- `σ` unchanged
- `warp_idx` tag (from UBO) so the host can demux

Boundary rejection (K·σ proximity to parallelogram edge) ALSO happens
on GPU — set feature's `valid` flag (a new field in the feature struct
or reuse `octave_idx` as a sentinel for "rejected"). The host filter
becomes a trivial mask.

### 7.2 Integration

Insert the back-projection dispatch at the end of the fused IMAS+detect
cmd buffer, after `recCopySIFTCountCmds`. Uses the per-slot UBO for
warp params.

### Acceptance criteria for Phase D

- 25-warp wallclock: ~200 ms.
- Feature positions match the host back-projection within < 0.01 px
  (just Float32 vs Float64 noise).
- Host-side feature processing becomes O(N_features) memcpy + filter,
  no per-feature trigonometry.

---

## 8. Test harness

### 8.1 Quick correctness check (post-Phase A)

```julia
# In BlobBoards kaimon session, after rebuild + Julia restart:
using BlobBoards, Colors, FixedPointNumbers, CUDA
using VisualGeometryFeatures: CudaBackend
using FileIO: load as load_image
include("examples/asift_gpu/imas_cpu.jl")
include("examples/asift_gpu/driver.jl")
using .ImasCpu, .AsiftGpu

image_path = joinpath(BlobBoards.data_dir(),
    "dataset_sonyfx3_tamron20mm_f8/jpeg/DSC02847.JPG")
img = Gray.(load_image(image_path))
H, W = size(img)

# Single-slot path (should be unchanged)
# … run asift_imas_gpu_cached as defined in dc846af test harness
# Expected: 86553 raw, 9720 after dark+NMS, 0.98 s
```

### 8.2 Recall validation against ImasCpu

Use the same comparison harness as `dc846af` — `KDTree` spatial match
at R=5 px against the all-CPU ImasCpu reference. Expected on the test
image:

```
CPU features:           3,431
GPU features (after NMS): 9,720
CPU recall (5px):       84.8%
σ ratio median:         1.000
Median position err:    1.81 px
```

Phases B/C should preserve these. Phase D may shift the median pos err
slightly (GPU back-projection vs CPU) but recall and σ ratio should be
within 0.5 pp.

### 8.3 Timing

```julia
# Single-slot baseline
@time gpu_feats = asift_imas_gpu_cached(imas25);

# Parallel (after Phase C)
@time gpu_feats = asift_imas_gpu_parallel(imas25, n_slots=4);
```

---

## 9. Risks & known gotchas

### 9.1 Vulkan instance singleton

VKS creates a global Vulkan instance via `vkenv_createInstance`. After a
`vksift_destroyInstance` call, the instance may not be re-creatable in
the same process. This bit us repeatedly in the session that landed
`dc846af`. Always test from a fresh Julia process; use kaimon's
`manage_repl command=restart` to force a clean state.

### 9.2 Per-warp size variation breaks pre-recorded cmd buffers

We hit this in the dc846af bugfix: passing different `(canvas_w,
canvas_h)` per warp causes `vksift_prepareSiftMemoryForDetection` to
reallocate the pyramid, which re-triggers `recordCommandBuffers`. For
parallel dispatch this is fatal — the cmd buffer being re-recorded
mid-flight would be invalid.

**Fix**: the parallel path must use a **stable canvas size across all
warps** (typically max_W_rot × max_H_sub). The IMAS warp's actual
sub-region size becomes a runtime parameter via the WarpParamsUBO. The
SIFT pyramid is sized once for the stable canvas; the quantize shader
writes only the valid sub-rectangle, fills the rest.

### 9.3 The `pending_warp_dirty` mechanism

Currently `vksift_detectFeaturesOnImas` sets `pending_warp_dirty=true`
to force a cmd-buffer re-record per call (so the QuantizePushConsts
get updated). With Phase B's UBO migration, this becomes unnecessary —
all warp params come from the UBO, not push consts. Remove the
`pending_warp_dirty` flag for the parallel path.

### 9.4 Single staging buffer

`image_staging_buffer` is shared across slots. It's only written by
host upload (regular `vksift_detectFeatures`), not by the IMAS waves,
so no contention. Confirm before Phase C that no parallel-wave code
path touches it.

### 9.5 Vulkan semaphores between waves

Successive waves don't need semaphores between them — fence-wait at
end of wave is sufficient. But subsequent submits should not signal/wait
on the same `end_of_detection_fence` simultaneously. Use a separate
fence pool sized for `n_pyramid_slots`.

### 9.6 GPU memory pressure

5 slots × ~3.5 GB ≈ 17.5 GB on a 24 GB 4090. Driver overhead + display
needs leave only ~5 GB headroom. **Test with n_pyramid_slots=2 first**;
ramp up after correctness validation. If memory pressure causes
allocation failures, fall back gracefully.

### 9.7 Float32 vs Float64 in back-projection

The host-side `tiltedcoor2imagecoor` does Float64 math. The GPU
back-projection shader will be Float32. Expect sub-0.01-px position
differences. This is fine for downstream `find_boards` (which has 1 px+
matching tolerances) but worth flagging in the commit.

---

## 10. Out of scope / future work

- **Multi-queue dispatch** (general + dedicated compute queue): possible
  refinement after Phase C. Could give ~1.3× extra concurrency on a
  4090 since it has both queues. Doesn't fundamentally change the design.
- **Adaptive `n_pyramid_slots`** based on input image size — for small
  images more slots fit in memory. Manual config for now.
- **GPU-side cross-warp NMS** — currently the 5×5 NMS runs on the host
  in Julia. Could move to GPU with the existing `CrossWarpSplat.comp`
  + `CrossWarpNms.comp` shaders (already in the repo). Probably saves
  ~30-50 ms more. Phase E if needed.

---

## 11. Open questions for the implementer

1. Does the 4090 actually achieve the 3-4× concurrency we're projecting,
   or are the SIFT shaders memory-bound enough that parallel pyramids
   don't help much? **Test after Phase C with n_slots ∈ {1, 2, 4, 5}**
   and observe wallclock — if scaling is sublinear, stop adding slots.

2. Is the WarpParamsUBO layout I sketched complete? Cross-check against
   the actual push_const layouts in every IMAS shader.

3. Should the back-projection's parallelogram boundary filter happen on
   GPU or be moved to the cross-warp NMS step? Either works; GPU side
   is slightly cheaper but couples to the back-projection shader.

---

## 12. Estimated calendar time

Working full-time on this:

- **Phase A**: 5–8 hours (large mechanical refactor with care needed
  for descriptor sets, image views, memory allocation alignment)
- **Phase B**: 6–8 hours (10 shader migrations + indirect dispatch
  plumbing + new fused cmd buffer recording)
- **Phase C**: 3–4 hours (public API + JL FFI + dispatch loop)
- **Phase D**: 2–3 hours (back-projection shader + integration)
- **Test & polish**: 2–3 hours

Total: **18–26 hours**, realistically a focused 3-day push.

Phase A alone is the riskiest — every per-pyramid resource touched, lots
of room for descriptor/barrier mistakes. Worth doing on its own day
with thorough single-slot validation before moving on.
