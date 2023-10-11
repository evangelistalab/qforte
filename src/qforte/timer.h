#include <chrono>
#include <vector>

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

    /// return the elapsed time in seconds
    void record(std::string name);

    /// returns a string representing a timings table
    std::string str_table(); 

  private:
    /// stores the time when this object is created
    std::chrono::high_resolution_clock::time_point start_;

    /// stores the timings and names
    std::vector<std::pair<std::string, double>> timings_;

    
};
