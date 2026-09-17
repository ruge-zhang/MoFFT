.text
.p2align 2

// Each kernel executes eight measured instructions per loop iteration.  The
// entry/exit transition is outside that count.  Independent destinations are
// selected by the ilp argument encoded in the function name.
.macro PROLOGUE
  stp d8, d9, [sp, #-16]!
  stp d10, d11, [sp, #-16]!
  stp d12, d13, [sp, #-16]!
  stp d14, d15, [sp, #-16]!
  smstart
  ptrue p0.b
  eor z0.b, z0.b, z0.b
  eor z1.b, z1.b, z1.b
  eor z2.b, z2.b, z2.b
  eor z3.b, z3.b, z3.b
  eor z4.b, z4.b, z4.b
  eor z5.b, z5.b, z5.b
  eor z6.b, z6.b, z6.b
  eor z7.b, z7.b, z7.b
.endm

.macro EPILOGUE
  smstop
  ldp d14, d15, [sp], #16
  ldp d12, d13, [sp], #16
  ldp d10, d11, [sp], #16
  ldp d8, d9, [sp], #16
  ret
.endm

.macro VEC_KERNEL name, op, lane, ilp
.globl _\name
.p2align 4
_\name:
  PROLOGUE
1:
  \op z0.\lane, p0/m, z0.\lane, z7.\lane
.if \ilp >= 2
  \op z1.\lane, p0/m, z1.\lane, z7.\lane
.else
  \op z0.\lane, p0/m, z0.\lane, z7.\lane
.endif
.if \ilp >= 4
  \op z2.\lane, p0/m, z2.\lane, z7.\lane
  \op z3.\lane, p0/m, z3.\lane, z7.\lane
.else
  \op z0.\lane, p0/m, z0.\lane, z7.\lane
  \op z1.\lane, p0/m, z1.\lane, z7.\lane
.endif
.if \ilp >= 8
  \op z4.\lane, p0/m, z4.\lane, z7.\lane
  \op z5.\lane, p0/m, z5.\lane, z7.\lane
  \op z6.\lane, p0/m, z6.\lane, z7.\lane
  \op z7.\lane, p0/m, z7.\lane, z6.\lane
.else
  \op z0.\lane, p0/m, z0.\lane, z7.\lane
  \op z1.\lane, p0/m, z1.\lane, z7.\lane
  \op z2.\lane, p0/m, z2.\lane, z7.\lane
  \op z3.\lane, p0/m, z3.\lane, z7.\lane
.endif
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm

.macro FMLA_KERNEL name, lane, ilp
.globl _\name
.p2align 4
_\name:
  PROLOGUE
1:
  fmla z0.\lane, p0/m, z6.\lane, z7.\lane
.if \ilp >= 2
  fmla z1.\lane, p0/m, z6.\lane, z7.\lane
.else
  fmla z0.\lane, p0/m, z6.\lane, z7.\lane
.endif
.if \ilp >= 4
  fmla z2.\lane, p0/m, z6.\lane, z7.\lane
  fmla z3.\lane, p0/m, z6.\lane, z7.\lane
.else
  fmla z0.\lane, p0/m, z6.\lane, z7.\lane
  fmla z1.\lane, p0/m, z6.\lane, z7.\lane
.endif
.if \ilp >= 8
  fmla z4.\lane, p0/m, z6.\lane, z7.\lane
  fmla z5.\lane, p0/m, z6.\lane, z7.\lane
  fmla z6.\lane, p0/m, z6.\lane, z7.\lane
  fmla z7.\lane, p0/m, z6.\lane, z7.\lane
.else
  fmla z0.\lane, p0/m, z6.\lane, z7.\lane
  fmla z1.\lane, p0/m, z6.\lane, z7.\lane
  fmla z2.\lane, p0/m, z6.\lane, z7.\lane
  fmla z3.\lane, p0/m, z6.\lane, z7.\lane
.endif
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm

.macro FMOPA_KERNEL name, lane, ilp
.globl _\name
.p2align 4
_\name:
  PROLOGUE
  zero {za}
1:
  fmopa za0.\lane, p0/m, p0/m, z0.\lane, z1.\lane
.if \ilp >= 2
  fmopa za1.\lane, p0/m, p0/m, z2.\lane, z3.\lane
.else
  fmopa za0.\lane, p0/m, p0/m, z2.\lane, z3.\lane
.endif
.if \ilp >= 4
  fmopa za2.\lane, p0/m, p0/m, z4.\lane, z5.\lane
  fmopa za3.\lane, p0/m, p0/m, z6.\lane, z7.\lane
.else
  fmopa za0.\lane, p0/m, p0/m, z4.\lane, z5.\lane
  fmopa za1.\lane, p0/m, p0/m, z6.\lane, z7.\lane
.endif
  fmopa za0.\lane, p0/m, p0/m, z0.\lane, z1.\lane
  fmopa za1.\lane, p0/m, p0/m, z2.\lane, z3.\lane
  fmopa za2.\lane, p0/m, p0/m, z4.\lane, z5.\lane
  fmopa za3.\lane, p0/m, p0/m, z6.\lane, z7.\lane
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm

.irp prec, fp32, fp64
  .ifc \prec,fp32
    .set lane_is_s, 1
  .endif
.endr

VEC_KERNEL mofft_mb_add_fp32_i1, fadd, s, 1
VEC_KERNEL mofft_mb_add_fp32_i2, fadd, s, 2
VEC_KERNEL mofft_mb_add_fp32_i4, fadd, s, 4
VEC_KERNEL mofft_mb_add_fp32_i8, fadd, s, 8
VEC_KERNEL mofft_mb_sub_fp32_i1, fsub, s, 1
VEC_KERNEL mofft_mb_sub_fp32_i2, fsub, s, 2
VEC_KERNEL mofft_mb_sub_fp32_i4, fsub, s, 4
VEC_KERNEL mofft_mb_sub_fp32_i8, fsub, s, 8
FMLA_KERNEL mofft_mb_fmla_v_fp32_i1, s, 1
FMLA_KERNEL mofft_mb_fmla_v_fp32_i2, s, 2
FMLA_KERNEL mofft_mb_fmla_v_fp32_i4, s, 4
FMLA_KERNEL mofft_mb_fmla_v_fp32_i8, s, 8
FMOPA_KERNEL mofft_mb_fmopa_fp32_i1, s, 1
FMOPA_KERNEL mofft_mb_fmopa_fp32_i2, s, 2
FMOPA_KERNEL mofft_mb_fmopa_fp32_i4, s, 4

VEC_KERNEL mofft_mb_add_fp64_i1, fadd, d, 1
VEC_KERNEL mofft_mb_add_fp64_i2, fadd, d, 2
VEC_KERNEL mofft_mb_add_fp64_i4, fadd, d, 4
VEC_KERNEL mofft_mb_add_fp64_i8, fadd, d, 8
VEC_KERNEL mofft_mb_sub_fp64_i1, fsub, d, 1
VEC_KERNEL mofft_mb_sub_fp64_i2, fsub, d, 2
VEC_KERNEL mofft_mb_sub_fp64_i4, fsub, d, 4
VEC_KERNEL mofft_mb_sub_fp64_i8, fsub, d, 8
FMLA_KERNEL mofft_mb_fmla_v_fp64_i1, d, 1
FMLA_KERNEL mofft_mb_fmla_v_fp64_i2, d, 2
FMLA_KERNEL mofft_mb_fmla_v_fp64_i4, d, 4
FMLA_KERNEL mofft_mb_fmla_v_fp64_i8, d, 8
FMOPA_KERNEL mofft_mb_fmopa_fp64_i1, d, 1
FMOPA_KERNEL mofft_mb_fmopa_fp64_i2, d, 2
FMOPA_KERNEL mofft_mb_fmopa_fp64_i4, d, 4
FMOPA_KERNEL mofft_mb_fmopa_fp64_i8, d, 8

.macro FMLAM_KERNEL name, lane, ilp, width
.globl _\name
.p2align 4
_\name:
  PROLOGUE
  zero {za}
  mov w8, #0
  mov w9, #1
  mov w10, #2
  mov w11, #3
1:
.if \width == 4
  fmla za.\lane[w8, #0], {z0.\lane-z3.\lane}, z4.\lane
.else
  fmla za.\lane[w8, #0], {z0.\lane-z1.\lane}, z4.\lane
.endif
.if \ilp >= 2
  .if \width == 4
    fmla za.\lane[w9, #0], {z0.\lane-z3.\lane}, z4.\lane
  .else
    fmla za.\lane[w9, #0], {z0.\lane-z1.\lane}, z4.\lane
  .endif
.else
  .if \width == 4
    fmla za.\lane[w8, #0], {z0.\lane-z3.\lane}, z4.\lane
  .else
    fmla za.\lane[w8, #0], {z0.\lane-z1.\lane}, z4.\lane
  .endif
.endif
.if \ilp >= 4
  .if \width == 4
    fmla za.\lane[w10, #0], {z0.\lane-z3.\lane}, z4.\lane
    fmla za.\lane[w11, #0], {z0.\lane-z3.\lane}, z4.\lane
  .else
    fmla za.\lane[w10, #0], {z0.\lane-z1.\lane}, z4.\lane
    fmla za.\lane[w11, #0], {z0.\lane-z1.\lane}, z4.\lane
  .endif
.else
  .if \width == 4
    fmla za.\lane[w8, #1], {z0.\lane-z3.\lane}, z4.\lane
    fmla za.\lane[w9, #1], {z0.\lane-z3.\lane}, z4.\lane
  .else
    fmla za.\lane[w8, #1], {z0.\lane-z1.\lane}, z4.\lane
    fmla za.\lane[w9, #1], {z0.\lane-z1.\lane}, z4.\lane
  .endif
.endif
  .if \width == 4
    fmla za.\lane[w8, #2], {z0.\lane-z3.\lane}, z4.\lane
    fmla za.\lane[w9, #2], {z0.\lane-z3.\lane}, z4.\lane
    fmla za.\lane[w10, #2], {z0.\lane-z3.\lane}, z4.\lane
    fmla za.\lane[w11, #2], {z0.\lane-z3.\lane}, z4.\lane
  .else
    fmla za.\lane[w8, #2], {z0.\lane-z1.\lane}, z4.\lane
    fmla za.\lane[w9, #2], {z0.\lane-z1.\lane}, z4.\lane
    fmla za.\lane[w10, #2], {z0.\lane-z1.\lane}, z4.\lane
    fmla za.\lane[w11, #2], {z0.\lane-z1.\lane}, z4.\lane
  .endif
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm

.macro ZAREAD_KERNEL name, lane, ilp
.globl _\name
.p2align 4
_\name:
  PROLOGUE
  zero {za}
  mov w12, #0
1:
  mov z0.\lane, p0/m, za0h.\lane[w12, 0]
.if \ilp >= 2
  mov z1.\lane, p0/m, za0h.\lane[w12, 0]
.else
  mov z0.\lane, p0/m, za0h.\lane[w12, 0]
.endif
.if \ilp >= 4
  mov z2.\lane, p0/m, za0h.\lane[w12, 0]
  mov z3.\lane, p0/m, za0h.\lane[w12, 0]
.else
  mov z0.\lane, p0/m, za0h.\lane[w12, 0]
  mov z1.\lane, p0/m, za0h.\lane[w12, 0]
.endif
  mov z0.\lane, p0/m, za0h.\lane[w12, 0]
  mov z1.\lane, p0/m, za0h.\lane[w12, 0]
  mov z2.\lane, p0/m, za0h.\lane[w12, 0]
  mov z3.\lane, p0/m, za0h.\lane[w12, 0]
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm

FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp32_i1, s, 1, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp32_i2, s, 2, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp32_i4, s, 4, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp32_i1, s, 1, 4
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp32_i2, s, 2, 4
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp32_i4, s, 4, 4
FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp64_i1, d, 1, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp64_i2, d, 2, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg2_fp64_i4, d, 4, 2
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp64_i1, d, 1, 4
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp64_i2, d, 2, 4
FMLAM_KERNEL mofft_mb_fmla_m_vg4_fp64_i4, d, 4, 4
ZAREAD_KERNEL mofft_mb_za_slice_fp32_i1, s, 1
ZAREAD_KERNEL mofft_mb_za_slice_fp32_i2, s, 2
ZAREAD_KERNEL mofft_mb_za_slice_fp32_i4, s, 4
ZAREAD_KERNEL mofft_mb_za_slice_fp64_i1, d, 1
ZAREAD_KERNEL mofft_mb_za_slice_fp64_i2, d, 2
ZAREAD_KERNEL mofft_mb_za_slice_fp64_i4, d, 4

.macro MIXED_KERNEL name, lane
.globl _\name
.p2align 4
_\name:
  PROLOGUE
  zero {za}
1:
  fadd z0.\lane, p0/m, z0.\lane, z7.\lane
  fmopa za0.\lane, p0/m, p0/m, z0.\lane, z1.\lane
  fadd z2.\lane, p0/m, z2.\lane, z7.\lane
  fmopa za1.\lane, p0/m, p0/m, z2.\lane, z3.\lane
  fadd z4.\lane, p0/m, z4.\lane, z7.\lane
  fmopa za2.\lane, p0/m, p0/m, z4.\lane, z5.\lane
  fadd z6.\lane, p0/m, z6.\lane, z7.\lane
  fmopa za3.\lane, p0/m, p0/m, z6.\lane, z7.\lane
  subs x0, x0, #1
  b.ne 1b
  mov x0, #8
  EPILOGUE
.endm
MIXED_KERNEL mofft_mb_mixed_add_fmopa_fp32_i4, s
MIXED_KERNEL mofft_mb_mixed_add_fmopa_fp64_i4, d

// Dedicated transition timing: x0 transitions, x0 returned.
.globl _mofft_mb_transition
.p2align 4
_mofft_mb_transition:
  // Streaming-mode transitions may invalidate the non-streaming vector
  // register contents. Preserve the ABI callee-saved low 64-bit lanes just as
  // the instruction kernels do; the C timing harness keeps scale constants in
  // this register bank across the call.
  stp d8, d9, [sp, #-16]!
  stp d10, d11, [sp, #-16]!
  stp d12, d13, [sp, #-16]!
  stp d14, d15, [sp, #-16]!
  mov x1, x0
1:
  smstart
  smstop
  subs x1, x1, #1
  b.ne 1b
  ldp d14, d15, [sp], #16
  ldp d12, d13, [sp], #16
  ldp d10, d11, [sp], #16
  ldp d8, d9, [sp], #16
  ret

.subsections_via_symbols
