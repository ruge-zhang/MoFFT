.text
.p2align 2

.globl _mofft_streaming_entry_f32
_mofft_streaming_entry_f32:
  stp x29, x30, [sp, #-80]!
  mov x29, sp
  stp d8, d9, [sp, #16]
  stp d10, d11, [sp, #32]
  stp d12, d13, [sp, #48]
  stp d14, d15, [sp, #64]
  smstart
  bl _mofft_execute_streaming_body_f32
  smstop
  ldp d8, d9, [sp, #16]
  ldp d10, d11, [sp, #32]
  ldp d12, d13, [sp, #48]
  ldp d14, d15, [sp, #64]
  ldp x29, x30, [sp], #80
  ret

.globl _mofft_streaming_entry_f64
_mofft_streaming_entry_f64:
  stp x29, x30, [sp, #-80]!
  mov x29, sp
  stp d8, d9, [sp, #16]
  stp d10, d11, [sp, #32]
  stp d12, d13, [sp, #48]
  stp d14, d15, [sp, #64]
  smstart
  bl _mofft_execute_streaming_body_f64
  smstop
  ldp d8, d9, [sp, #16]
  ldp d10, d11, [sp, #32]
  ldp d12, d13, [sp, #48]
  ldp d14, d15, [sp, #64]
  ldp x29, x30, [sp], #80
  ret
