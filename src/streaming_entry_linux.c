/* Linux SME1 entry; see docs/linux-sme1-port.md for the build contract.
 * The public interface is non-streaming and private-ZA. Let the compiler
 * implement the AAPCS64 TPIDR2/lazy-save contract and streaming transitions;
 * do not substitute a bare SMSTART / call / SMSTOP assembly wrapper here.
 * Neither public FFT semantics nor the streaming-body mathematics change.
 */
#include "mofft.h"
#include <arm_sme.h>

void mofft_execute_streaming_body_f32(const mofft_plan_f32 *plan,
                                     const mofft_complex_f32 *input,
                                     mofft_complex_f32 *output)
    __arm_streaming __arm_inout("za");
void mofft_execute_streaming_body_f64(const mofft_plan_f64 *plan,
                                     const mofft_complex_f64 *input,
                                     mofft_complex_f64 *output)
    __arm_streaming __arm_inout("za");

__attribute__((target("sme")))
__arm_new("za") __arm_locally_streaming
void mofft_streaming_entry_f32(const mofft_plan_f32 *plan,
                               const mofft_complex_f32 *input,
                               mofft_complex_f32 *output) {
  mofft_execute_streaming_body_f32(plan, input, output);
}

__attribute__((target("sme")))
__arm_new("za") __arm_locally_streaming
void mofft_streaming_entry_f64(const mofft_plan_f64 *plan,
                               const mofft_complex_f64 *input,
                               mofft_complex_f64 *output) {
  mofft_execute_streaming_body_f64(plan, input, output);
}
