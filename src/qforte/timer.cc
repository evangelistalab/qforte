#include <timer.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <sstream>

void local_timer::reset() {
     start_ = std::chrono::high_resolution_clock::now();
}

double local_timer::get() {
    auto duration = std::chrono::high_resolution_clock::now() - start_;
    return std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
}

void local_timer::record(std::string name) {
    auto duration = std::chrono::high_resolution_clock::now() - start_;
    double time = std::chrono::duration_cast<std::chrono::duration<double>>(duration).count();
    timings_.push_back(std::make_pair(name, time));
}

std::string local_timer::str_table() {
    std::stringstream result;

    size_t max = 0;
    double total_time = 0.0;

    for (const auto& entry : timings_) {
        max = std::max(max, entry.first.length());
        total_time += entry.second;
    }

    max = std::max(max, static_cast<size_t>(10));

    result << std::setw(max) << "Process name" << std::setw(max) << "Time (s)" << std::setw(max) << "Percent" << "\n";
    result << std::setw(max) << "=============" << std::setw(max) << "=============" << std::setw(max) << "=============" << "\n";

    for (const auto& entry : timings_) {
        double percent = (entry.second / total_time) * 100.0;
        result << std::setw(max) << entry.first << std::fixed << std::setprecision(4) << std::setw(max) << entry.second
               << std::fixed << std::setprecision(2) << std::setw(max) << percent << "\n";
    }

    result << "\n";

    result << std::setw(max) << "Total Time" << std::fixed << std::setprecision(4) << std::setw(max) << total_time
           << std::fixed << std::setprecision(2) << std::setw(max) << 100.0 << "\n";

    return result.str();
}




