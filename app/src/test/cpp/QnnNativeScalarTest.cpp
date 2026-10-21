// Compile independently of Qualcomm SDK:
//   c++ -std=c++17 -Wall -Wextra -Werror -fsanitize=undefined -I app/src/main/cpp/include
//       app/src/test/cpp/QnnNativeScalarTest.cpp -o /tmp/qnn_scalar_test && /tmp/qnn_scalar_test
#include "QnnNativeScalar.hpp"
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>

int main() {
    int32_t signedValue = 1;
    assert(localdream::qnn::finite_round_to_integer(0.f, signedValue) && signedValue == 0);
    assert(localdream::qnn::finite_round_to_integer(999.f, signedValue) && signedValue == 999);
    assert(localdream::qnn::finite_round_to_integer(-999.f, signedValue) && signedValue == -999);
    assert(localdream::qnn::finite_round_to_integer(127.4f, signedValue) && signedValue == 127);
    assert(!localdream::qnn::finite_round_to_integer(std::numeric_limits<float>::infinity(), signedValue));
    assert(!localdream::qnn::finite_round_to_integer(std::numeric_limits<float>::quiet_NaN(), signedValue));
    assert(!localdream::qnn::finite_round_to_integer(2147483648.f, signedValue));

    uint32_t unsignedValue = 0;
    assert(localdream::qnn::finite_round_to_integer(123.f, unsignedValue) && unsignedValue == 123);
    assert(!localdream::qnn::finite_round_to_integer(-1.f, unsignedValue));
    assert(!localdream::qnn::finite_round_to_integer(4294967296.f, unsignedValue));
}
