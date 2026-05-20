#ifndef VKSIFT_WARP_UBO_H
#define VKSIFT_WARP_UBO_H
//
// WarpParamsUBO — host↔GLSL contract for the IMAS+detect parallel-pyramid
// pipeline. Single uniform buffer, bound at descriptor set 1, binding 0 in
// every IMAS-chain shader (AffineWarp, GaussBlur1DStorage, FinvsplineRow,
// FinvsplineCol, FprojCubicY, FprojBilinearY, QuantizeF32ToInput) plus the
// detector's on-IMAS path (AffineWarp + Quantize).
//
// All scalars pack tightly at 4-byte alignment in std140 (no implicit
// padding inside a flat block of scalars — std140 alignment only kicks in
// for vec3/vec4/struct members). The fields below are organized in 16-byte
// rows for readability; the explicit _padN floats keep the C and GLSL
// layouts in sync if the compiler ever inserts implicit padding.
//
// Phase B-2 consumer: a single UBO write per command-buffer recording is
// reused by all 5 IMAS shaders for one warp. Future phases will reuse the
// same struct for the GPU back-projection shader (warp_idx slot).
//

#include <stdint.h>
#include <vulkan/vulkan.h>

// =============================================================================
// SlotDispatchBuffer — per-slot indirect-dispatch group counts for the fused
// IMAS + Quantize chain (Phase B-3). Lives in
// `vksift_SiftPyramidSlot::dispatch_buffer` (host-mapped). Host writes the
// VkDispatchIndirectCommand triples (x, y, z group counts) before each warp
// submission; the pre-recorded fused command buffer reads them via
// vkCmdDispatchIndirect.
//
// SIFT-detect dispatches (PreBlur1D, AffineWarp on warped_input, scale-space,
// DoG, ExtractKeypoints, CopySIFTCount) are NOT here — their group counts
// derive from the stable curr_input_image_* canvas, which doesn't change
// across IMAS warps in a wave. Those stay direct dispatches.
//
// Workgroup sizes (must match shader local_size_*):
//   AffineWarp.comp           local_size_x=8,  local_size_y=8
//   GaussBlur1DStorage.comp   local_size_x=8,  local_size_y=8
//   FinvsplineRow.comp        local_size_x=64
//   FinvsplineCol.comp        local_size_x=64
//   FprojCubicY/BilinearY.comp local_size_x=8, local_size_y=8
//   QuantizeF32ToInput.comp   local_size_x=8,  local_size_y=8
// =============================================================================
typedef struct
{
  VkDispatchIndirectCommand affinewarp;       // (ceil(W_rot/8), ceil(H_rot/8), 1)
  VkDispatchIndirectCommand gaussblur;        // (ceil(W_rot/8), ceil(H_rot/8), 1)
  VkDispatchIndirectCommand finvspline_row;   // (ceil(H_rot/64), 1, 1)
  VkDispatchIndirectCommand finvspline_col;   // (ceil(W_rot/64), 1, 1)
  VkDispatchIndirectCommand fproj;            // (ceil(W_rot/8), ceil(H_sub/8), 1)
  VkDispatchIndirectCommand quantize;         // (ceil(canvas_w/8), ceil(canvas_h/8), 1)
  VkDispatchIndirectCommand seed_from_input;  // (ceil(oct0_w/8), ceil(oct0_h/8), 1)
} SlotDispatchBuffer;

typedef struct
{
  // [bytes  0..31] — AffineWarp inverse matrix + OOB fills (32 B)
  float a11, a12, a13;
  float a21, a22, a23;
  // AffineWarp OOB sample fill. IMAS path: 0.5 (matches imas_cpu.jl FROT_FILL).
  // Detector AffineWarp path: detector->pending_warp_fill (caller-settable).
  float fill_value;
  // Fproj{Cubic,Bilinear}Y OOB fill — separate from AffineWarp fill because
  // IMAS pushes 0.5 to AffineWarp and 0.0 to Fproj. Unifying them broke
  // bit-exact output at boundary taps. Default in Phase B: 0.0 for Fproj.
  float fproj_bg_value;

  // [bytes 32..63] — dimension fields (32 B)
  // W_rot / H_rot semantics by shader:
  //   AffineWarp.comp   : (output_width, output_height) of the rotated canvas
  //   GaussBlur1D.comp  : (in_w, in_h) of the storage image being blurred
  //   Finvspline{Row,Col}.comp : (width, height) of the IIR domain
  //   FprojCubicY.comp / FprojBilinearY.comp : (input_width, input_height) =
  //       (W_rot, H_rot) — and the output dims are (W_rot, H_sub).
  // canvas_w/h, valid_w/h, warp_idx are for QuantizeF32ToInput / back-project.
  uint32_t W_rot;
  uint32_t H_rot;
  uint32_t H_sub;     // FprojCubicY output_height = floor(H_rot / t_factor)
  uint32_t canvas_w;  // QuantizeF32ToInput canvas (= curr_input_image_width)
  uint32_t canvas_h;  // QuantizeF32ToInput canvas (= curr_input_image_height)
  uint32_t valid_w;   // QuantizeF32ToInput valid sub-region (= W_rot)
  uint32_t valid_h;   // QuantizeF32ToInput valid sub-region (= H_sub)
  uint32_t warp_idx;  // Tag stamped onto back-projected features (Phase D)

  // [bytes 64..95] — float scalars (32 B)
  float sigma_aa;       // = 0.8 * sqrt(t^2 - 1); 0 at identity tilt
  float t_factor;       // Tilt factor used by Fproj{Cubic,Bilinear}Y
  float quantize_fill;  // Value to fill outside the IMAS-written sub-region
  float gauss_dir_x;    // GaussBlur1D direction X component (0 or 1)
  float gauss_dir_y;    // GaussBlur1D direction Y component (0 or 1)
  float _pad1;
  float _pad2;
  float _pad3;

  // [bytes 96..143] — Phase D back-projection params (48 B).
  // Host-filled per warp by vksift_fillFusedWarpState; consumed by
  // BackProjectFeatures.comp to (a) reject features whose K·σ·σ_max boundary
  // pad overlaps the parallelogram edge and (b) back-project tilted-frame
  // (x, y) → input-frame (x, y) via the 2×3 affine [bp_a11 bp_a12 bp_a13;
  // bp_a21 bp_a22 bp_a23].
  //   A_inv = R(-φ) · diag(1, t) · R(φ)      (2×2 symmetric)
  //   bp_a11 = cφ² + t·sφ²,   bp_a12 = bp_a21 = (t-1)·cφ·sφ
  //   bp_a22 = sφ² + t·cφ²
  // Center offset (so input_xy = A_inv · tilted_xy + (c - A_inv · c)):
  //   bp_a13 = cx − (bp_a11·cx + bp_a12·cy)
  //   bp_a23 = cy − (bp_a21·cx + bp_a22·cy)
  // where c = ((W-1)/2, (H-1)/2). σ_max of A_inv is max(1, t) = t for t ≥ 1.
  float    bp_a11, bp_a12, bp_a13;
  float    bp_a21, bp_a22, bp_a23;
  float    bp_sigma_max;        // σ_max of A_inv (= max(1, t))
  float    bp_boundary_K;       // K in K·σ·σ_max boundary pad (default 3.0)
  uint32_t bp_input_W;          // original input image width  (for rect bounds)
  uint32_t bp_input_H;          // original input image height
  uint32_t bp_is_identity;      // 1 for the t=1, φ=0 warp (skip boundary check)
  uint32_t bp_nb_octaves;       // mem->curr_nb_octaves — shader walks this many sections

  // [bytes 144..159] — Symmetric inverse-tilt SHAPE matrix (16 B, 1 padding).
  // The shape matrix S of an ASIFT feature in the INPUT frame is σ · A_inv
  // where A_inv = R(-φ)·diag(1, t)·R(φ) is symmetric. BackProjectFeatures.comp
  // multiplies by σ and writes s_xx, s_xy, s_yy onto each feature record so
  // downstream consumers receive an input-frame ellipse rather than a
  // tilted-frame circle. The position back-projection (bp_a*) uses a different
  // (asymmetric) matrix because the IMAS forward warp itself is asymmetric.
  float    bp_shape_a11;        // cφ² + t·sφ²
  float    bp_shape_a12;        // (t-1)·cφ·sφ   (=bp_shape_a21 by symmetry)
  float    bp_shape_a22;        // sφ² + t·cφ²
  float    _bp_shape_pad;
} WarpParamsUBO;

#endif // VKSIFT_WARP_UBO_H
