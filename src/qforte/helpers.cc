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
    if (std::imag(value) == 0.0){
      return fmt::format("{:+f}", std::real(value));
    }
    if (std::real(value) == 0.0){
      return fmt::format("{:+f}j", std::imag(value));
    }
    return fmt::format("{:+f} {:+f}i", std::real(value), std::imag(value));
}

// TODO(Tyler): Need to expose and need a test case
// More of a helper funcitn may be better to put elsewhere
int reverse_bubble_list(std::vector<std::vector<int>>& arr) {
    int larr = arr.size();
    int swap_count = 0;
    for (int i = 0; i < larr; ++i) {
        bool swapped = false;
        for (int j = 0; j < larr - i - 1; ++j) {
            if (arr[j][0] < arr[j + 1][0]) {
                std::swap(arr[j], arr[j + 1]);
                swapped = true;
                ++swap_count;
            }
        }
        if (!swapped) {
            break;
        }
    }
    return swap_count;
}

// Function that performs parity sort and returns the number of swaps
int parity_sort(std::vector<size_t>& arr) {
    size_t arr_size = arr.size();
    std::vector<std::pair<size_t, size_t>> parr(arr_size);

    // Create pairs of (parity, value) for each element in arr
    for (size_t i = 0; i < arr_size; ++i) {
        parr[i] = std::make_pair(arr[i] % 2, arr[i]);
    }

    int swap_count = 0;

    for (size_t i = 0; i < arr_size; ++i) {
        bool swapped = false;
        for (size_t j = 0; j < arr_size - i - 1; ++j) {
            if (parr[j].first > parr[j + 1].first) {
                std::swap(parr[j], parr[j + 1]);
                swapped = true;
                swap_count++;
            }
        }
        if (!swapped) {
            break;
        }
    }

    for (size_t i = 0; i < arr_size; ++i) {
        arr[i] = parr[i].second;
    }

    return swap_count;
}