.text
.p2align 2

// Enter streaming mode once around a uniform candidate wrapper. Arguments:
// body, input, row_stride, twiddles, twiddle_stride, output, batch.
// Saves/restores the SVE callee-saved registers z8-z15 (and p4-p15) inside
// streaming mode, so a non-streaming caller does not need to spill them
// (which would emit `addvl` — illegal on Apple Silicon outside streaming).
.globl _mofft_candidate_entry
_mofft_candidate_entry:
  stp x29, x30, [sp, #-16]!
  mov x29, sp
  mov x9, x0
  mov x0, x1
  mov x1, x2
  mov x2, x3
  mov x3, x4
  mov x4, x5
  mov x5, x6
  smstart
  addvl sp, sp, #-16
  str z8,  [sp, #0, mul vl]
  str z9,  [sp, #1, mul vl]
  str z10, [sp, #2, mul vl]
  str z11, [sp, #3, mul vl]
  str z12, [sp, #4, mul vl]
  str z13, [sp, #5, mul vl]
  str z14, [sp, #6, mul vl]
  str z15, [sp, #7, mul vl]
  str p4,  [sp, #8, mul vl]
  str p5,  [sp, #9, mul vl]
  str p6,  [sp, #10, mul vl]
  str p7,  [sp, #11, mul vl]
  str p8,  [sp, #12, mul vl]
  str p9,  [sp, #13, mul vl]
  str p10, [sp, #14, mul vl]
  str p11, [sp, #15, mul vl]
  blr x9
  ldr p11, [sp, #15, mul vl]
  ldr p10, [sp, #14, mul vl]
  ldr p9,  [sp, #13, mul vl]
  ldr p8,  [sp, #12, mul vl]
  ldr p7,  [sp, #11, mul vl]
  ldr p6,  [sp, #10, mul vl]
  ldr p5,  [sp, #9, mul vl]
  ldr p4,  [sp, #8, mul vl]
  ldr z15, [sp, #7, mul vl]
  ldr z14, [sp, #6, mul vl]
  ldr z13, [sp, #5, mul vl]
  ldr z12, [sp, #4, mul vl]
  ldr z11, [sp, #3, mul vl]
  ldr z10, [sp, #2, mul vl]
  ldr z9,  [sp, #1, mul vl]
  ldr z8,  [sp, #0, mul vl]
  addvl sp, sp, #16
  smstop
  ldp x29, x30, [sp], #16
  ret

.subsections_via_symbols
