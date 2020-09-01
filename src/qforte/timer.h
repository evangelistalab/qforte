#include <chrono>

/**
 * @brief A timer class that returns the elapsed time
 */
class local_timer {
  public:
    local_timer() : start_(std::chrono::high_resolution_clock::now()) {}

    /// reset the timer
    void reset();

    /// return the elapsed time in seconds
    double get();

  private:
    /// stores the time when this object is created
    std::chrono::high_resolution_clock::time_point start_;
};
