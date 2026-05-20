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
stash@{0}  phase-c-multi-queue WIP   (Phase E — next)
```

(Old `phase-d-broken` and `phase-d-approach1-and-diagnosis` stashes are
obsolete — Phase D landed via a different fix. Drop them with
`git stash drop` once you confirm `git stash list` matches.)

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

## If you only have time for one thing next session

Phase E (multi-queue). The Phase E stash (`git stash list` — was
`stash@{1}` originally, numbering may have shifted) bumps
`nb_async_compute_queues 0 → 1` and routes half each wave's slots to
the async-compute queue, but per-slot resources are
`VK_SHARING_MODE_EXCLUSIVE` owned by the graphics family so the
compute-queue writes produce undefined contents. Fix: recreate the ~10
per-slot resources (`sift_buffer_arr[s]`,
`sift_count_staging_buffer_arr[s]`, `slots[s].input_image`,
`slots[s].rotated_image`, `cached_input_image`, etc.) with
`VK_SHARING_MODE_CONCURRENT` listing both
`general_queues_family_idx` and `async_compute_queues_family_idx`.

Expected payoff per plan §10: ~1.3× on the n_slots>1 path. At
n_slots=2 that should put 1920²×25 warps near 210 ms (from 276 ms),
finally beating the n_slots=1 result and getting close to the plan's
200 ms target.
