#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

namespace localdream::qnn {
// SDK-independent checked float -> integer conversion. SDXL's INT32 timestep
// and other discrete QNN tensors must never overflow or accept NaN / infinity.
template <typename Int>
bool finite_round_to_integer(float source, Int& destination) {
    static_assert(std::numeric_limits<Int>::is_integer, "integer type required");
    const double rounded = std::nearbyint(static_cast<double>(source));
    if (!std::isfinite(rounded) ||
        rounded < static_cast<double>(std::numeric_limits<Int>::lowest()) ||
        rounded > static_cast<double>(std::numeric_limits<Int>::max())) return false;
    destination = static_cast<Int>(rounded);
    return true;
}
} // namespace localdream::qnn
