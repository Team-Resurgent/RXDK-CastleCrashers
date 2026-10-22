// C library functions the generated code needs that the SDK's libc lacks.
#include <cmath>

// Round half to even (SIMDe's _mm_round_ps to nearest).
extern "C" float roundevenf(float x) {
    float whole = floorf(x);
    float part = x - whole;
    if (part > 0.5f || (part == 0.5f && fmodf(whole, 2.0f) != 0.0f))
        whole += 1.0f;
    return whole;
}
