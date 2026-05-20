# Parallel-Pyramid IMAS+Detect — Status & Handoff

Continuation notes for the next session picking up
[PARALLEL_PYRAMID_PLAN.md](PARALLEL_PYRAMID_PLAN.md). Read that plan first
for context; this file is the after-action report for the implementation
attempt that landed Phases A–C and bounced off Phase D + multi-queue.

Repo: `/home/jbpritts/src/VulkanSift`, branch `asift-batch`.
Julia driver: `/home/jbpritts/src/BlobBoards.jl/examples/asift_gpu/driver.jl`.

---

## TL;DR

- **Phase A + B + C landed and are bit-exact** against the pre-refactor baseline
  for every IMAS-25 warp. The parallel infrastructure (per-slot pyramid memory,
  per-slot descriptor sets, fused IMAS+detect cmd buffer, vkQueueSubmit waves,
  JL FFI surface) is solid.
- **No wallclock win materialised from Phase C.** The 4090's single Vulkan
  queue serialises independent cmd-buffer submissions; n_slots=4 ≈ n_slots=1
  on the IMAS-25 schedule.
- **Phase D (GPU back-projection) plumbed but stashed.** Hits an unresolved
  NVIDIA-specific quirk where the back-project shader's set=1 descriptor
  binding reads wave-0 UBO values for every wave. Three fix attempts failed.
- **Multi-queue stashed.** Per-slot resources are `VK_SHARING_MODE_EXCLUSIVE`;
  cross-queue writes need ownership transfer or `CONCURRENT` recreate. Plan
  §10 estimated ~1.3× from this anyway — limited upside.

---

## What landed (commits on `asift-batch`)

| Commit  | Phase   | Summary |
|---------|---------|---------|
| `b881ea4` | A       | `vksift_SiftPyramidSlot[VKSIFT_MAX_PYRAMID_SLOTS]`; setup/destroy loop over slots; detector + IMAS read `mem->slots[0].X` |
| `cc6ed4e` | B-1     | Per-slot `warp_params_ubo` (256 B) + `dispatch_buffer` (256 B), host-mapped |
| `dc511d9` | B-2     | 7 IMAS shaders push_const → UBO at descriptor set 1; `WarpParamsUBO` struct in `sift_warp_ubo.h` |
| `13564d3` | B-3     | Fused IMAS+detect cmd buffer per slot, indirect dispatch on IMAS chain dims, `vksift_dispatchFusedImasWarpForSlot`, `vksift_jl_detect_fused_imas` |
| `080acf8` | C-1     | Per-slot descriptor sets for every pipeline binding per-pyramid views: detector preblur/affinewarp/blur/dog/downsample/quantize/extractkpts/orientation/descriptor/rgba/rgb + IMAS warp/blur/finvspline/fproj |
| `a0ad847` | C-2+C-3 | `nb_sift_buffer ≥ nb_pyramid_slots` auto-bump; `vksift_dispatchParallelIMAS` + `vksift_jl_dispatch_parallel_imas` + JL FFI feature cache; driver.jl `asift_gpu_detect_parallel` |
| `cb2c3dc` | (mem)   | Rotated/tilted scratch sized at `sqrt(W²+H²)` instead of `(W+H)` — saves ~800 MB per slot at 7084² (3.2 GB at n_slots=4) |

## Tags

```
pre-phase-a-pyramid-slots → phase-a-complete → phase-b-complete → phase-c-complete
```

Rollback any phase with `git reset --hard <tag>`.

## Stashes

```
stash@{0}  phase-d-broken
stash@{1}  phase-c-multi-queue WIP
```

Both have detailed reproduction notes in their stash messages.

---

## Correctness validation

Every phase was checked bit-exact against the prior baseline using the
existing driver:

```julia
using BlobBoards, FileIO, Colors, FixedPointNumbers
include("examples/asift_gpu/imas_cpu.jl")
include("examples/asift_gpu/driver.jl")
using .AsiftGpu
img_path = joinpath(homedir(), ".julia", "artifacts",
    "42fe55c9c1812c79f0acd650026a0a5c6336b74a", "blob_board_336_0997a.png")
img_u8 = collect(reinterpret(UInt8,
    map(g -> N0f8(Colors.gray(g)), Gray.(FileIO.load(img_path)))))

# Expected at every phase tag (post-A, post-B, post-C):
#   identity warp: 15644 raw features
#   IMAS-25 sum_raw: 259480
#   IMAS-25 sum_kept: 182764
```

Phase-B-3 additionally verified `vksift_jl_detect_fused_imas` matches the
non-fused `vksift_jl_run_imas + vksift_jl_detect_on_imas` pair bit-exact
at fixed canvas across `t ∈ {1.0, 1.41, 2.0, 2.82, 4.0}` (per plan §9.2 —
the canvas must be stable across warps).

Phase-C-3 verified `asift_gpu_detect_parallel` output matches across
`n_slots ∈ {1, 2, 4}` — same `sum_raw = 170383` (full IMAS chain) /
`sum_kept = 59635` on 7084², `111948 raw = 111948 kept` on 1920² before
Phase D's boundary filter.

---

## Performance reality

| Setup                                  | Wallclock | Notes                                          |
|----------------------------------------|-----------|------------------------------------------------|
| Phase A baseline (asift_gpu_detect)    | ~650 ms (7084²) | Pre-refactor reference                  |
| Phase B-3 fused, n_slots=1, 5 warps   | 141 ms (1920²) | 1.18× vs non-fused IMAS+detect chain    |
| Phase C parallel, n_slots=1, 25 warps | 531 ms (1920²) | Baseline                                |
| Phase C parallel, n_slots=4, 25 warps | 570 ms (1920²) | **Slower** than n_slots=1               |
| Phase C parallel, n_slots=2, 25 warps | 2290 ms (7084²) | Slower than n_slots=1's 1692 ms        |
| Phase C parallel, n_slots≥3 at 7084²  | OOM at init  | Per-slot ~4.4 GB after sqrt fix         |

### Why no Phase C speedup

Plan §11 anticipated this as an open question. The empirical answer:

1. **Single-queue serialisation.** All wave's cmd buffers go to
   `detector->general_queue`. NVIDIA's driver executes them sequentially
   on that engine even though the cmd buffers touch disjoint memory.
2. **GPU SM saturation.** One warp's IMAS chain + ~6-octave SIFT pyramid
   (~132 dispatches/warp) keeps the 4090 fully busy. Additional concurrent
   warps queue behind, no overlap.
3. **Per-wave fence overhead.** Each wave pays a `vkResetFences` +
   `vkQueueSubmit` + `vkWaitForFences` + per-slot count download
   round-trip.

To actually parallelise across slots you need a real second hardware
queue (compute family separate from graphics) AND fix the sharing-mode
issue (stash@{1}). Even then, plan §10 estimated only ~1.3×.

### What CAN still buy time

| Source                                    | Estimated gain |
|-------------------------------------------|----------------|
| Phase D (GPU back-projection)             | ~100-115 ms saved at 1920² (currently host) — IF the UBO freeze gets cracked |
| Multi-queue (Phase E)                     | 1.3× per plan §10                            |
| Fewer octaves (cap at 5)                  | ~10-20% wallclock, slight recall hit         |
| Fewer scales per octave                   | ~33% blur dispatches saved, bigger quality hit |
| Drop the lowest octave entirely           | small absolute win                            |

`detection_only=true` is already set (no orientation, no descriptor).

---

## Phase D — the UBO freeze (stash@{0})

### Symptom

`vksift_jl_detect_fused_imas` produced ~111948 "kept" features at 1920²
n_slots=1 — i.e. zero boundary rejection — when the shader's identity
flag should reject most warp-1+ features.

Diagnostic shader code wrote `data[i].intensity =
1000*ubo.bp_is_identity + ubo.warp_idx` to each feature; reading back
per-warp showed **every warp returned 1000.0**. That decomposes to
`bp_is_identity=1` and `warp_idx=0` — wave-0's values, frozen.

### What's confirmed working

- The shader IS running (forcing unconditional reject gave kept=0).
- One UBO field, `bp_input_W` (= 1920 = `W`), DOES read correctly because
  the host writes the same value every wave so we can't distinguish
  "reads fresh" from "reads wave-0" for that field alone.
- The IMAS-chain shaders earlier in the same fused cmd buffer DO read
  fresh per-wave UBO values — Phase B-3's fused-vs-non-fused bit-exact
  match across `t ∈ {1, 1.41, ...}` confirms this. Different t produces
  different `sigma_aa`, different `t_factor` etc., and the output reflects it.

### What's NOT the cause

- ❌ UBO struct layout / std140 packing: `bp_input_W` reads correctly at
  its expected offset (128); offsets of `warp_idx` (60) and `bp_is_identity`
  (136) follow the same scalar packing.
- ❌ Cmd-buffer reuse caching the descriptor lookup: forcing full
  `recordCommandBuffers` (writeDescriptorSets + recordCommandBuffers)
  per wave inside `vksift_dispatchParallelIMAS` did not change the result.
- ❌ Host writes not flushing: HOST_COHERENT memory is used; an explicit
  HOST_WRITE → UNIFORM_READ + INDIRECT_COMMAND_READ pipeline barrier
  added at the start of the fused cmd buffer did not change the result.
- ❌ UBO vs SSBO caching: rebinding the same buffer as `readonly buffer`
  at std430 (with `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` added on
  the slot's warp_params_ubo, separate `backproject_warp_sbo_*`
  layout/pool/sets) did not change the result.

### Theories that haven't been ruled out

1. **Pipeline barrier scoping**. The IMAS-chain → Quantize → SIFT-detect
   chain has many `vkCmdPipelineBarrier` calls in between, each
   restricted to specific images/buffers. None of them touch the
   `warp_params_ubo`. NVIDIA may have collapsed the UBO read across the
   whole cmd buffer into a single early load that goes to a memory
   address resolved at submission time but whose VALUE is bound at
   first execution.
2. **Specialization across re-submits**. Some drivers cache compiled
   compute pipeline state per "execution context" of a cmd buffer. With
   the same cmd buffer re-submitted, the back-project dispatch's UBO
   load might get resolved-once and then re-used. The IMAS shaders escape
   this because their dispatch dims come from indirect-dispatch buffers
   (force fresh evaluation?) — back-project's dispatch dims are direct
   `vkCmdDispatch(1u, 1u, 1u)` per (slot, octave).

### What to try next

In priority order:

1. **Per-wave allocation of a fresh cmd buffer** specifically for the
   back-project dispatch, kept in a small ring buffer of N pre-allocated
   `VkCommandBuffer` slots in `detector->backproject_cmd_pool`. Submit
   it as a SECOND `VkSubmitInfo` after the fused (IMAS+detect) cmd buffer
   in the same `vkQueueSubmit` array. The fused cmd buffer reuse stays;
   only the back-project cmd buffer is re-recorded each wave. This
   sidesteps whatever the cache issue is by always using a fresh
   `VkCommandBuffer`.
2. **Indirect dispatch for back-project** — bake the `(1, 1, 1)` group
   counts into the slot's `dispatch_buffer` (already host-mapped). The
   theory is that indirect-dispatch buffers trigger NVIDIA's "I need to
   re-evaluate" path on cmd buffer reuse; this is the difference between
   the IMAS chain (works) and back-project (broken).
3. **Validation layers run in verbose mode**. The repo builds with
   `#ifndef NDEBUG` validation; route the log through `stderr` directly
   (currently stdout goes through kaimon which strips it) or run the
   `test_affine_warp` standalone C test under `vulkan-tools/vkconfig` to
   see what the validation layers say about descriptor-set state for the
   back-project pipeline.
4. **Try a different device path for testing.** The agentic test was
   only on an NVIDIA 4090. A second GPU (or `LIBGL_ALWAYS_SOFTWARE`-style
   software Vulkan) might reveal whether this is an NVIDIA driver bug
   or a real Vulkan-spec violation in our code.

The stashed code is functional in every other respect — the back-project
math, boundary check, descriptor allocation, dispatch wiring, and Julia
driver simplification are all correct. Once the UBO freeze is unblocked,
unstashing and re-validating should be ~30 min.

To unstash and continue:

```bash
git stash apply stash@{0}        # restores Phase D files (no auto-commit)
cd /home/jbpritts/src/VulkanSift/build && make -j
cd /home/jbpritts/src/VulkanSift && \
  gcc -shared -fPIC -O2 -I include -o build/libvksift_jl.so vksift_jl.c \
      -L build -lvulkansift -Wl,-rpath,'$ORIGIN'
```

Then restart the kaimon REPL and exercise the `asift_gpu_detect_parallel`
path on 1920² — expected `sum_kept ≈ 59635` on the 7084² test image
(matches Julia-back-projection output bit-exact ±0.01 px per plan §9.7).
With the bug, you'll see `sum_kept = sum_raw` instead.

---

## Multi-queue (Phase E) — the sharing-mode issue (stash@{1})

### Symptom

`n_slots=2` produced 9737 features (vs the expected 111948); slots
routed to the async-compute queue produced garbage downloads.

### Root cause

Plan §10 mentions multi-queue is a follow-up. The implementation:

1. Bumped `vksift_createInstance`'s `nb_async_compute_queues` 0 → 1.
2. Added `async_compute_command_pool` + per-slot mirror
   `fused_imas_detect_command_buffer_compute[VKSIFT_MAX_PYRAMID_SLOTS]`
   in the new compute pool.
3. In `vksift_dispatchParallelIMAS`, split each wave's slots half-and-half
   between `detector->general_queue` and
   `detector->dev->async_compute_queues[0]` with two separate fences.

What went wrong: every per-slot resource that the compute-queue cmd
buffer writes — `sift_buffer_arr[s]`, `sift_count_staging_buffer_arr[s]`,
`slots[s].input_image`, `slots[s].rotated_image`, etc. — was created
with `VK_SHARING_MODE_EXCLUSIVE` (and owned by `general_queues_family_idx`).
Per Vulkan spec, the compute-queue write into a graphics-family-owned
buffer/image produces *undefined* contents without an explicit ownership
transfer barrier. Also `cached_input_image` is read by both queues
without ownership transfer — same problem.

### What to try next

The clean fix is to recreate every per-slot SIFT buffer + count staging
buffer + slot image + `cached_input_image` with
`VK_SHARING_MODE_CONCURRENT` listing both
`general_queues_family_idx` and `async_compute_queues_family_idx`. About
10 allocations in `sift_memory.c`'s `setupOneSlot` + `setupStaticObjectsAndMemory`
+ `vksift_createSiftMemory`.

Touching the sharing mode on the SIFT staging buffer (host-readable) is
also needed since the compute-queue cmd buffer writes counts there too.

A simpler diagnostic first: pin slot 0 to general queue, slot 1 to
compute queue, then bypass the per-slot SIFT buffer for slot 1 (just
verify the IMAS chain produces correct rotated_image content on the
compute queue). That isolates whether the issue is the SIFT buffer alone
or every cross-queue read.

Realistic gain from multi-queue per plan §10: ~1.3×. Not worth the
complexity unless Phase D is also fixed (Phase D saves ~100 ms host-side
deterministically; multi-queue × Phase D × phase-B-3's 1.18× could
plausibly bring 1920²×25 warps from ~530 ms to ~250 ms).

To unstash:

```bash
git stash apply stash@{1}        # restores multi-queue files
# then go fix the EXCLUSIVE → CONCURRENT sharing-mode issue in sift_memory.c
```

---

## File index — what each commit touches

```
src/vulkansift/sift_memory.h           Phase A + B-1 (slot struct + UBO/dispatch fields)
src/vulkansift/sift_memory.c           Phase A + B-1 + C-2 + sqrt sizing
src/vulkansift/sift_detector.h         Phase A + B-3 + C-1 + C-3
src/vulkansift/sift_detector.c         Every phase
src/vulkansift/sift_imas.h             Phase A (slot lookup) + B-2 (UBO contract) + B-3 (refresh decl)
src/vulkansift/sift_imas.c             Phase A + B-2 + B-3 (refresh impl)
src/vulkansift/sift_warp_ubo.h         B-2 (struct definition) + B-3 (SlotDispatchBuffer)
src/vulkansift/shaders/*.comp          B-2 (push_const → UBO migration on 7 shaders)
src/vulkansift/vulkansift.c            Config defaults; B-3 + C-3 entry points
src/vulkansift/vkenv/vulkan_utils.{h,c} B-2 (createComputePipeline2 helper)
include/vulkansift/vulkansift.h        B-3 + C-3 public API
vksift_jl.{h,c}                        B-3 + C-3 FFI surface (3 new symbols → 15 total)
CMakeLists.txt                         (only stash@{0} touched — added BackProjectFeatures.comp)
```

Julia driver:

```
examples/asift_gpu/driver.jl  added vks_init `n_pyramid_slots` kwarg
                              + asift_gpu_detect_parallel function
```

---

## Test image / setup notes

- Test image: `~/.julia/artifacts/42fe55c9.../blob_board_336_0997a.png` (7084×7084 synthetic blob board). Resized to 1920×1920 for memory-friendly tests; the 7084² original fits at `n_slots ≤ 2`.
- IMAS-25 schedule (`AsiftGpu.imas_tilts_25()`) at `t ∈ {1, √2, 2, 2√2, 4}` × 5 phi values.
- Vulkan instance singleton (plan §9.1): once `vksift_destroyInstance` runs, recreation in the same process fails. Restart kaimon between test runs via `mcp__kaimon__manage_repl restart`.
- Build: `cd build && make -j`, then `gcc -shared -fPIC -O2 -I include -o build/libvksift_jl.so vksift_jl.c -L build -lvulkansift -Wl,-rpath,'$ORIGIN'` (the FFI .so is NOT in CMake; rebuild manually after any vksift_jl.c or libvulkansift.so change).
- After rebuild, restart kaimon for the new symbol table.

---

## If you only have time for one thing next session

Fix the Phase D UBO freeze (stash@{0}) using approach #1 (separate
per-wave back-project cmd buffer) — it sidesteps the cache issue without
needing to diagnose the root cause. That alone delivers the ~100 ms
deterministic savings the plan attributed to Phase D, while keeping the
fused cmd buffer's amortised recording. If that lands, the whole
pipeline at 1920²×25 warps should drop from 530 ms → 410-430 ms,
nontrivial but well short of the 200 ms plan target.
