# Parallel-Pyramid IMAS+Detect — Status & Handoff

Continuation notes for the next session picking up
[PARALLEL_PYRAMID_PLAN.md](PARALLEL_PYRAMID_PLAN.md). Read that plan first
for context; this file is the after-action report for the implementation
that landed Phases A–D. Phase E (multi-queue) is still stashed.

Repo: `/home/jbpritts/src/VulkanSift`, branch `asift-batch`.
Julia driver: `/home/jbpritts/src/BlobBoards.jl/examples/asift_gpu/driver.jl`.

---

## TL;DR

- **Phase A + B + C + D landed and are bit-exact** against the Julia
  back-projection reference for every IMAS-25 warp. The parallel
  infrastructure (per-slot pyramid memory, per-slot descriptor sets, fused
  IMAS+detect+back-project cmd buffer, vkQueueSubmit waves, JL FFI surface)
  is solid.
- **Phase D delivered its ~100 ms savings.** 1920²×25 warps: 531 ms (pre-D)
  → 276 ms (post-D, n_slots=1, min over 5 runs). Host-side trigonometry +
  parallelogram check are gone.
- **No wallclock win materialised from Phase C.** The 4090's single Vulkan
  queue serialises independent cmd-buffer submissions; n_slots=4 (332 ms
  min) actually runs *slower* than n_slots=1 because the wave's serialised
  fence/download overhead dominates without true queue parallelism.
- **The "Phase D UBO freeze" was a misdiagnosis.** The original stash
  blamed an NVIDIA UBO cache bug, but the actual issue was that
  `vksift_dispatchParallelIMAS` computed `warp_idx = base + s` where
  `base` was relative to the call, and the JL FFI chunked by `n_slots`
  (calling the dispatch once per warp for `n_slots=1`). The host wrote
  `warp_idx=0`, `bp_is_identity=1` for every wave — the GPU read what
  was written. Fix in commit-pending: add `warp_idx` to
  `vksift_WarpSpec`, propagate global index through the FFI, derive
  `bp_is_identity` from `(t_factor, theta_rad)` intrinsically so any
  caller works without a per-call convention.
- **Multi-queue (Phase E) still stashed.** Per-slot resources are
  `VK_SHARING_MODE_EXCLUSIVE`; cross-queue writes need ownership transfer
  or `CONCURRENT` recreate. Plan §10 estimated ~1.3× upside — now the
  next obvious lever.

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
| _pending_ | D       | GPU back-projection (`BackProjectFeatures.comp`) + boundary filter inside the fused IMAS+detect cmd buffer; `vksift_WarpSpec.warp_idx` for global schedule index; `bp_is_identity` derived intrinsically from `(t_factor, theta_rad)` |

## Tags

```
pre-phase-a-pyramid-slots → phase-a-complete → phase-b-complete → phase-c-complete
```

Rollback any phase with `git reset --hard <tag>`. Phase D doesn't have
a tag yet — add one after the commit lands.

## Stashes

```
stash@{0}  phase-e-wip            (latest Phase E attempt — see "Phase E attempt" below)
stash@{1}  phase-c-multi-queue    (original Phase E scaffold from before the wip)
```

Phase E details: see the "Phase E attempt" section near the bottom of
this doc. The wip stash got further than the original scaffold (it adds
CONCURRENT sharing on per-slot resources, a graphics→compute semaphore,
and a defensive HOST→COMPUTE barrier) but still hits a wall on the
compute-queue dispatch producing zero / garbage features for everything
beyond the first identity warp.

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

Phase D verified `asift_gpu_detect_parallel` output matches across
`n_slots ∈ {1, 2, 4}` on 1920² — same `sum_raw = 111950 / sum_kept =
36924` — and across `n_slots ∈ {1, 2}` on 7084² — same `sum_raw =
170383 / sum_kept = 59635`, the plan §9.7 reference value.

---

## Performance reality

| Setup                                  | Wallclock | Notes                                          |
|----------------------------------------|-----------|------------------------------------------------|
| Phase A baseline (asift_gpu_detect)    | ~650 ms (7084²) | Pre-refactor reference                  |
| Phase B-3 fused, n_slots=1, 5 warps   | 141 ms (1920²) | 1.18× vs non-fused IMAS+detect chain    |
| Phase C parallel, n_slots=1, 25 warps | 531 ms (1920²) | Pre-Phase-D                             |
| **Phase D landed, n_slots=1, 25 warps** | **276 ms (1920²)** | **Min over 5; median 295 ms**       |
| Phase D landed, n_slots=4, 25 warps   | 332 ms (1920²) | Min over 5; n_slots=4 still slower than n_slots=1 (single-queue serialisation) |
| Phase C parallel, n_slots=2, 25 warps | 2290 ms (7084²) | Pre-Phase-D                             |
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
issue (stash@{2}; numbering shifted after the Phase D rewrite —
double-check `git stash list`). Plan §10 estimated ~1.3× from this.

### What CAN still buy time

| Source                                    | Estimated gain |
|-------------------------------------------|----------------|
| Multi-queue (Phase E)                     | 1.3× per plan §10                            |
| Fewer octaves (cap at 5)                  | ~10-20% wallclock, slight recall hit         |
| Fewer scales per octave                   | ~33% blur dispatches saved, bigger quality hit |
| Drop the lowest octave entirely           | small absolute win                            |

`detection_only=true` is already set (no orientation, no descriptor).
Phase D's host-trigonometry savings (~100-115 ms at 1920²) is already
banked.

---

## Phase D — what actually happened

### The misdiagnosis

The original Phase D stash (`stash@{1}: phase-d-broken`) shipped a long
list of "this didn't fix it" notes blaming an NVIDIA UBO cache freeze.
The symptom was real — every IMAS-25 warp reported `sum_kept == sum_raw`
on the 1920² test image, meaning the K·σ·σ_max boundary filter never
fired — but the explanation was wrong.

The stash's smoking-gun diagnostic encoded
`data[i].intensity = 1000*ubo.bp_is_identity + ubo.warp_idx` and saw
1000.0 returned for every warp. The author read that as "the UBO is
frozen at wave 0's contents" and tried four cache-flushing workarounds:
HOST→COMPUTE pipeline barrier on the UBO, SSBO rebinding, per-wave
re-record of the fused cmd buffer, and an `INDIRECT_COMMAND_READ`
dependency. None worked, because none of them touched the actual cause.

### The actual cause

`vksift_dispatchParallelIMAS` computed `warp_idx = base + s` inside its
inner wave loop, where `base` is the loop counter for *that call*, not
a global schedule offset. The JL FFI (`vksift_jl_dispatch_parallel_imas`)
chunks the dispatch by `nb_pyramid_slots` so each FFI call carries one
internal wave; with `n_slots = 1`, that means 25 separate dispatch calls
for the IMAS-25 schedule, each with `n_warps = 1`. Inside every one of
those calls the inner loop reduced to `base = 0, s = 0`, so the host
wrote `ubo.warp_idx = 0` and `ubo.bp_is_identity = 1` *every* wave.

The GPU was reading exactly what the CPU wrote. No cache bug.

A targeted diagnostic confirmed the asymmetry the stash author had
missed: at the same offset the host re-wrote per warp, `ubo.bp_a11`
read fresh per warp (different values for different φ), while
`ubo.warp_idx` and `ubo.bp_is_identity` stayed at the values *for warp
zero* — exactly the values the FFI/dispatcher kept stamping into them
on every call.

### The fix

Three coupled changes:

1. Extend `vksift_WarpSpec` (public API in
   `include/vulkansift/vulkansift.h`) with a `uint32_t warp_idx` field
   so callers can stamp the global schedule index. Callers that don't
   need it can still leave it 0; the value is only consumed for
   diagnostics now (see point 3).
2. `vksift_dispatchParallelIMAS` passes `warps[base+s].warp_idx`
   through to `vksift_fillFusedWarpState` instead of computing
   `base + s` internally.
3. **`vksift_fillFusedWarpState` derives `bp_is_identity` from
   `(t_factor, theta_rad)` directly** — `(t_factor == 1.0f) &&
   (theta_rad == 0.0f)` — rather than from `warp_idx`. This makes the
   identity check intrinsic to the warp transform instead of dependent
   on a "schedule index 0 = identity" convention; both the parallel and
   serial entry points now work correctly without per-caller bookkeeping.

The cleanup that came with the fix: the SSBO descriptor set, the
`backproject_warp_sbo_*` layout/pool/sets, the
`VK_BUFFER_USAGE_STORAGE_BUFFER_BIT` on `slots[s].warp_params_ubo`, the
`HOST→COMPUTE` barrier at the head of the fused cmd buffer, and the
per-wave force-record diagnostic in `vksift_dispatchParallelIMAS` are
all gone. The BP shader binds the existing per-slot UBO at
`set=1 binding=0` as a regular UBO — same descriptor set the IMAS
shaders use.

### Validation

| Resolution | n_slots | sum_raw | sum_kept | Notes                       |
|------------|---------|---------|----------|-----------------------------|
| 1920²      | 1       | 111950  | 36924    |                             |
| 1920²      | 2       | 111950  | 36924    |                             |
| 1920²      | 4       | 111950  | 36924    | Wave-count-invariant        |
| 7084²      | 1       | 170383  | 59635    | Matches plan §9.7 exactly   |
| 7084²      | 2       | 170383  | 59635    |                             |

Per-warp counts make sense: the identity warp (t=1, φ=0) keeps every
feature; high-tilt warps (t=5.18) reject most of theirs to the
parallelogram boundary; identical counts across `n_slots` runs.

### Lessons for future "stale-UBO" diagnoses

- The first thing to check when fields read stale is **what the host
  is actually writing** for the field — print the bytes from the
  host-mapped UBO pointer right before submission and compare against
  the dispatcher's understanding of "what wave is this".
- A diagnostic that reads fields next to each other from the SAME
  buffer (`bp_a11` vs `warp_idx` in this case) trivially refutes
  whole-buffer cache theories. The stash's diagnostic only read uint
  fields, which all happened to be ones the dispatcher was writing
  with constant per-call values.
- "It worked on the IMAS chain shaders" doesn't mean the UBO is read
  fresh — it can also mean those shaders only consume *float* fields
  that the host happens to write fresh per warp. None of the IMAS
  shaders read `warp_idx`; the field is just there for layout
  compatibility.

### Phase D perf, measured

1920² × IMAS-25, single warp per FFI chunk, median of 5 runs after warmup:

| Setup                    | min ms | median ms |
|--------------------------|--------|-----------|
| Pre-Phase-D, n_slots=1   | —      | 531       |
| Post-Phase-D, n_slots=1  | 276    | 295       |
| Post-Phase-D, n_slots=4  | 332    | 346       |

`n_slots=4` is slower than `n_slots=1` for the same reason Phase C
didn't help: a single-queue 4090 serialises the cmd buffers anyway and
the per-wave fence/feature-count download overhead grows with wave size.
This is the next thing Phase E should fix.

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
vksift_jl.{h,c}                        B-3 + C-3 FFI surface (3 new symbols → 15 total); D adds warp_idx propagation through the WarpSpec array
CMakeLists.txt                         D adds BackProjectFeatures.comp to VULKANSIFT_LIB_SHADERS
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

## Phase E status — partial commit + remaining stash

A short session re-attempted Phase E with the Khronos validation layer
loaded (`~/.local/share/vulkan/explicit_layer.d/VkLayer_khronos_validation.json`
installed with an absolute `library_path`, so no `VK_ADD_LAYER_PATH`
env-dance needed for future runs). Validation immediately surfaced the
primary blocker as **VUID-VkBufferMemoryBarrier-None-09050** — the
legacy `recBufferOwnershipTransferCmds` barriers had non-IGNORED
queue-family indices on what was (intended to be) a CONCURRENT SIFT
buffer. That problem is now fixed and committed on `asift-batch`; the
remaining Phase E machinery (multi-queue dispatch logic, async-compute
cmd pool, cross-queue semaphore) is still stashed.

### What landed in this session (committed)

- `multi_queue_share_info(...)` helper in `sift_memory.c`. Returns
  CONCURRENT across whichever of (general, async-compute, async-transfer)
  the device exposes, EXCLUSIVE if only general is available.
- `sift_buffer_arr`, `sift_count_staging_buffer_arr`, and
  `match_output_buffer` switched to CONCURRENT via the helper.
- `recBufferOwnershipTransferCmds` in **both** `sift_detector.c` and
  `sift_matcher.c` neutralized to no-ops. With the SIFT buffer
  CONCURRENT, ownership transfers are unnecessary AND illegal per
  VUID-VkBufferMemoryBarrier-None-09050. Cross-queue execution + memory
  dependency on the legacy single-queue path still holds because
  `vkQueueSubmit`'s semaphore signal/wait establishes both per the
  Vulkan synchronization spec.

Validation on the n_slots=1 parallel path is now clean (the remaining
VUID-vkCmdBlitImage-srcImage-00219 + VUID-vkDestroyDevice-device-05137
violations are pre-existing, not Phase E). Functional regression check:
`sum_raw=111948 sum_kept=36923` at 1920² × IMAS-25, matching the
Phase D baseline of 111950/36924 to within rounding-level drift.

### Pre-existing bugs that validation also surfaced (not fixed)

- `warped_input_image` (output of AffineWarp) is created with
  `SAMPLED | STORAGE | TRANSFER_DST` but missing `TRANSFER_SRC_BIT`.
  `recScaleSpaceConstructionCmds` blits FROM it at octave 0 →
  VUID-vkCmdBlitImage-srcImage-00219 fires 20+ times per detection.
  Harmless on NVIDIA (the blit still works) but a spec violation —
  add `TRANSFER_SRC_BIT` to the create flags in `sift_memory.c`.
- Multiple `VkPipelineLayout`/`VkPipeline`/`VkDescriptorSet*` leaks at
  `vkDestroyDevice` (object-tracking VUID-vkDestroyDevice-device-05137).
  Some pipeline created during init isn't destroyed in the matching
  cleanup path. Worth a sweep when convenient.

### What's still in `stash@{0}` (phase-e-wip — numbering may shift)

Everything Phase E-specific that doesn't make sense without the rest
of Phase E being functional:

1. **Per-slot resources beyond the SIFT buffer made CONCURRENT.** The
   committed change covers `sift_buffer_arr` /
   `sift_count_staging_buffer_arr` / `match_output_buffer` (the buffers
   that actually triggered the validation error). The stash extends
   CONCURRENT to per-slot images (`input_image`, `rotated_image`,
   `tilted_image`, `blur_tmp_image`, `octave_image`, `dog_image`,
   `warped_input_image`, `blurred_input_image`, `rgba_input_image`),
   per-slot `warp_params_ubo` / `dispatch_buffer`, `cached_input_image`,
   `indirect_orientation_dispatch_buffer`,
   `indirect_descriptor_dispatch_buffer`, and the per-slot RGB input
   buffer. None of those are touched by the compute queue at HEAD —
   they only matter when the multi-queue dispatcher lands.

2. **Async-compute command pool + per-slot cmd buffer mirror.**
   `detector->async_compute_command_pool` allocated against
   `async_compute_queues_family_idx`, plus
   `fused_imas_detect_command_buffer_compute[VKSIFT_MAX_PYRAMID_SLOTS]`
   allocated from it. `recordCommandBuffers` records the same content
   into both the general-pool and compute-pool cmd buffers so the
   dispatcher can submit half of every wave on each queue.

3. **Phase E dispatcher logic in `vksift_dispatchParallelIMAS`.** Wave
   split (`wave_g = (wave+1)/2`, `wave_c = wave - wave_g`), two-fence
   reset/submit/wait, `parallel_compute_start_semaphore` cross-queue
   handshake, defensive `HOST→COMPUTE` barrier at the head of the
   fused cmd buffer.

4. **Sync objects + destructor cleanup for the above.**
   `end_of_detection_fence_compute`, `parallel_compute_start_semaphore`,
   and corresponding `VK_NULL_SAFE_DELETE` calls in
   `vksift_destroySiftDetector`.

### Two bugs the validation session found in the stashed code (still TODO)

After landing the CONCURRENT-sharing + barrier-removal cleanup, replaying
the stash + the validation layer surfaced two further Phase E bugs that
need fixing before multi-queue can actually work:

1. **`vkCmdBlitImage` on the compute-queue cmd buffer.**
   `recScaleSpaceConstructionCmds` emits a `vkCmdBlitImage` at octave 0
   (`warped_input_image` → `octave_image_arr[0]` with `VK_FILTER_LINEAR`).
   Blits require the GRAPHICS queue-family capability bit; the
   async-compute family on RTX 4090 (family index 2) lacks it
   (`VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT | VK_QUEUE_SPARSE_BINDING_BIT`).
   VUID-vkCmdBlitImage-commandBuffer-cmdpool fires when the compute-pool
   mirror cmd buffer is recorded.
   **Fix paths:**
   - Replace with `vkCmdCopyImage` when octave 0 dims == input dims
     (the common case — `upsample=false`), with a compute-shader
     downsample path for the upsample case.
   - OR write a compute-shader linear-resample that runs on either queue.

2. **`parallel_compute_start_semaphore` double-signal**
   (VUID-vkQueueSubmit-pSignalSemaphores-00067). Each
   `vksift_dispatchParallelIMAS` call signals once at the top and the
   first compute submit waits once — on paper balanced — but validation
   sees the semaphore signaled twice without an intervening wait.
   The cause was not pinned down in the validation session. Likely
   suspects: the FFI's `wave_size = n_slots` chunking means the
   "last chunk" may have `n_warps == 1` which sets
   `dispatch_uses_compute = false` (no signal), but other chunks with
   `n_warps == n_slots == 2` always signal — and somewhere the wait
   isn't crediting the signal. Could also be cross-queue tracking in
   the validation layer.
   **Fix path:** instrument with `Gate.stash` / a print of which
   submits actually signal vs wait per call, OR replace the binary
   semaphore with a timeline semaphore (multi-signal-safe by spec).

Replay path for the next session: `git stash apply stash@{0}` (verify
slot number with `git stash list` — the cleanup above didn't touch the
stash, but stash numbering shifts as you make new ones), then tackle
those two bugs.

### Original stash notes (pre-cleanup)

The diff includes the original `stash@{1}: phase-c-multi-queue` scaffold
(async-compute pool + per-slot compute-pool cmd buffers + dedicated
fence + wave-splitting in `vksift_dispatchParallelIMAS`) plus three
additional pieces that take it further but not all the way:

1. **`VK_SHARING_MODE_CONCURRENT` for everything the compute queue
   touches.** A `multi_queue_share_info` helper in `sift_memory.c` fills
   `(sharing_mode, queue_family_count, queue_family_indices)` for callers,
   returning EXCLUSIVE on devices without a dedicated compute queue
   family so we don't pay CONCURRENT's modest driver-side cost on
   hardware that can't use it. Applied to every per-slot resource in
   `setupOneSlot`, plus `cached_input_image`, `sift_buffer_arr[s]`,
   `sift_count_staging_buffer_arr[s]`, and the indirect orientation /
   descriptor dispatch buffers.

2. **Cross-queue handshake semaphore.** New
   `detector->parallel_compute_start_semaphore`. At the top of
   `vksift_dispatchParallelIMAS`, an empty signal-only submit on
   `general_queue` signals the semaphore; the first compute-queue wave
   waits on it at `COMPUTE_SHADER_BIT`. Reason: `cached_input_image`
   was last written by the graphics queue in `vks_detect`; the host's
   `vkWaitForFences` between the two calls gives the host visibility
   but does NOT establish a graphics→compute *GPU-side* dependency,
   even with CONCURRENT sharing. Subsequent compute submissions on the
   same queue see cached_input_image transitively through submission
   order, so the semaphore is one-shot per `vksift_dispatchParallelIMAS`
   call.

3. **Defensive `HOST→COMPUTE` barrier at the head of every fused cmd
   buffer.** `HOST_WRITE → UNIFORM_READ | SHADER_READ |
   INDIRECT_COMMAND_READ` covering both the slot's
   `warp_params_ubo` and `dispatch_buffer` host writes that happen
   immediately before each wave's submit.

### Symptom that's still unresolved

With all three pieces in: **only the first identity warp (warp_idx=0,
t=1, φ=0) on slot 0 of the general queue in the very first wave
returns features.** Specifically:

- `n_slots=1`: still works perfectly (276 ms, sum_kept=36924 on 1920²,
  59635 on 7084²) — but n_slots=1 never exercises the compute queue.
- `n_slots=2` or `n_slots=4`: only warp 1 (identity, slot 0, general
  queue) returns 9738 features. Every other warp returns 0 features —
  including warps on the GENERAL queue in waves 2+. Sometimes returns
  garbage `octave_idx` (`32687`, `1768715626`) — uninitialised memory
  semantics.
- Forcing `wave_g=0, wave_c=wave` (everything on compute) gives 9738 for
  warp 1, 0 for everything else. So the compute queue produces
  *something* for the identity warp's lightweight IMAS work but fails on
  anything heavier — or fails the second time the cmd buffer is
  submitted, regardless of which queue.

That last symptom — wave-2 general-queue failures — strongly suggests
something past CONCURRENT sharing is leaking state between submissions.
Possible suspects:

- The shared indirect orientation / descriptor dispatch buffers
  (CONCURRENT now, but still single-writer-multi-slot — `slot 0` and
  `slot 1` both `vkCmdFillBuffer` the same per-octave offsets).
  Pre-Phase-E this raced harmlessly because both queues wrote the same
  values; Phase E might expose a real ordering issue.
- The indirect dispatch buffer state (host-written between waves) not
  re-read by NVIDIA's compute queue on subsequent submissions even with
  the HOST→COMPUTE barrier — same pattern as the misdiagnosed Phase D
  "UBO freeze" but actually on the compute path.
- A compute-queue-specific Vulkan-spec violation flagged by the
  Khronos validation layer — which we couldn't get loaded under
  kaimon's Julia (the validation layer's `.so` is now in the pixi env
  at `.pixi/envs/default/lib/libVkLayer_khronos_validation.so` but the
  Vulkan loader needs `VK_ADD_LAYER_PATH` + `LD_LIBRARY_PATH` set
  **before** Julia starts; in-Julia `ENV[]` is too late since libvulkan
  is already `dlopen`'d).

### Infrastructure landed (for the next attempt)

These two pieces are committed on `asift-batch` as Phase-E-prep and are
no-ops at runtime when validation isn't loaded:

- **`VK_EXT_debug_utils` messenger** in `vkenv_createInstance`
  (`src/vulkansift/vkenv/vulkan_device.c`). Registers a callback that
  prints validation-layer warnings/errors to **stderr** (kaimon forwards
  stderr, strips stdout). When `VK_LAYER_KHRONOS_validation` is loaded
  by the loader, every validation message lands directly in the kaimon
  output. When the layer isn't loaded, the messenger registration is a
  no-op.
- **`vulkan-validation-layers` in `pixi.toml`** (`pixi.lock` updated).
  Installs the Khronos validation layer + its `.so` into the pixi env.

### How to actually get validation output flowing next time

The validation layer load can't be triggered from inside a running
Julia process. Either:

```bash
# Option A: launch the kaimon-Julia process with the right env exported.
export VK_ADD_LAYER_PATH="$PWD/.pixi/envs/default/share/vulkan/explicit_layer.d"
export LD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib:$LD_LIBRARY_PATH"
# … start kaimon as usual …
```

```bash
# Option B: copy / symlink the layer's .so into a path already on
# the loader's runtime search list, e.g. /usr/local/lib (needs sudo)
# or one of the colon-separated entries in `cat /etc/ld.so.conf.d/*`.
```

```bash
# Option C: build VulkanSift with -DCMAKE_BUILD_TYPE=Debug so
# `vksift_loadVulkan`'s `#ifndef NDEBUG` branch fires AND ship the
# validation layer system-wide. Slightly heavier (debug build) but
# avoids the per-launch env-var dance.
```

The `vksift_loadVulkan` source path that requests
`VK_LAYER_KHRONOS_validation` lives under `#ifndef NDEBUG`. For a
release build, manually un-gate that block (or temporarily edit
`vulkansift.c`) before re-running.

### Expected payoff if Phase E lands

Per plan §10: ~1.3× on the n_slots>1 path. At n_slots=2 that should put
1920²×25 warps near 210 ms (from 276 ms), finally beating the n_slots=1
result and getting close to the plan's 200 ms target.

### If you only have time for one thing next session

Get `VK_LAYER_KHRONOS_validation` loaded (Option A above is easiest),
unstash `phase-e-wip`, run the n_slots=2 test, and read the validation
output. The messenger code already routes those messages to stderr;
kaimon will capture them. With validation telling us exactly which
spec rule we're tripping (queue-family transition? synchronization
scope? layout transition?), Phase E should land in another short
session.
