// Each active receiver owns a contiguous slice of changed CSR edges. The
// unchanged 25.4M-edge base operator is evaluated separately by its provider.
// Keeping E/I channels separate preserves the visual kernel's sign rule.
extern "C" __global__ void owner_overlay_delta(
    int active_count,
    const int* __restrict__ receiver,
    const long long* __restrict__ begin,
    const long long* __restrict__ end,
    const int* __restrict__ source,
    const double* __restrict__ weight_before,
    const double* __restrict__ weight_after,
    const double* __restrict__ release_before,
    const double* __restrict__ release_after,
    const double* __restrict__ caps,
    const unsigned char* __restrict__ visual,
    double visual_scale,
    bool visual_sources_connected,
    double* __restrict__ delta)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= active_count) return;
    const int row = receiver[k];
    double ordinary = 0.0;
    double excitatory = 0.0;
    double inhibitory = 0.0;
    const bool is_visual = visual[row] != 0;
    for (long long e = begin[k]; e < end[k]; ++e) {
        const int pre = source[e];
        const double old_value = weight_before[e] * release_before[pre];
        const double new_value = weight_after[e] * release_after[pre];
        if (is_visual) {
            const double old_scaled = old_value * visual_scale;
            const double new_scaled = new_value * visual_scale;
            excitatory += fmax(new_scaled, 0.0) - fmax(old_scaled, 0.0);
            inhibitory += fmax(-new_scaled, 0.0) - fmax(-old_scaled, 0.0);
        } else if (visual_sources_connected || visual[pre] == 0) {
            ordinary += (new_value - old_value) * caps[pre];
        }
    }
    delta[k] = ordinary;
    delta[active_count + k] = excitatory;
    delta[2 * active_count + k] = inhibitory;
}
