#include <timer.h>

void local_timer::reset() { start_ = std::chrono::high_resolution_clock::now(); }

double local_timer::get() {
    auto duration = std::chrono::high_resolution_clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}
