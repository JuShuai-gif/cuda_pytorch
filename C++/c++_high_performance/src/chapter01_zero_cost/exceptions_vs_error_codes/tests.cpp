#include <cstdio>
#include <stdexcept>

#include "baseline.hpp"
#include "optimized.hpp"
#include "test_utils.hpp"

int main() {
    // Success path: both produce the same result.
    int out = 0;
    CHP_CHECK(chp::evc::divide_checked(100, 5, out) ==
              chp::evc::DivideError::none);
    CHP_CHECK(out == 20);
    CHP_CHECK(chp::evc::divide_throwing(100, 5) == 20);

    // Division by zero, error-code style: caller must check the return value.
    CHP_CHECK(chp::evc::divide_checked(100, 0, out) ==
              chp::evc::DivideError::division_by_zero);

    // Division by zero, exception style: caller must handle the exception.
    bool threw = false;
    try {
        (void)chp::evc::divide_throwing(100, 0);
    } catch (const std::runtime_error&) {
        threw = true;
    }
    CHP_CHECK(threw);

    // Truncating integer division behaves identically in both styles.
    out = 0;
    CHP_CHECK(chp::evc::divide_checked(-7, 3, out) ==
              chp::evc::DivideError::none);
    CHP_CHECK(out == chp::evc::divide_throwing(-7, 3));

    return chp::test_summary("exceptions_vs_error_codes");
}
