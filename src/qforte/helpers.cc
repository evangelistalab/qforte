#include "fmt/format.h"

#include "helpers.h"

std::string join(const std::vector<std::string>& vec_str, const std::string& sep) {
    if (vec_str.size() == 0)
        return std::string();

    std::string ss;

    std::for_each(vec_str.begin(), vec_str.end() - 1, [&](const std::string& s) { ss += s + sep; });
    ss += vec_str.back();

    return ss;
}

std::string to_string(std::complex<double> value) {
    if (std::imag(value) == 0.0) {
        return fmt::format("{:+f}", std::real(value));
    }
    if (std::real(value) == 0.0) {
        return fmt::format("{:+f}j", std::imag(value));
    }
    return fmt::format("{:+f} {:+f}i", std::real(value), std::imag(value));
}