#include <stdexcept>

#include "fmt/format.h"

#include "gate.h"

Gate make_gate(std::string type, size_t target, size_t control, double parameter) {
    // using namespace std::complex_literals;
    std::complex<double> onei(0.0, 1.0);
    if (target == control) {
        if (type == "X") {
            std::complex<double> gate[4][4]{
                {0.0, 1.0},
                {1.0, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Y") {
            std::complex<double> gate[4][4]{
                {0.0, -onei},
                {+onei, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Z") {
            std::complex<double> gate[4][4]{
                {+1.0, 0.0},
                {0.0, -1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "H") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +c},
                {+c, -c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "R") {
            std::complex<double> tmp = onei * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "Rx") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = onei * std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {-b, +a},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "Ry") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {+b, +a},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "Rz") {
            std::complex<double> tmp_a = -onei * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = onei * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {a, 0.0},
                {0.0, b},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "V") {
            std::complex<double> a = onei * 0.5 + 0.5;
            std::complex<double> b = -onei * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {+a, +b},
                {+b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "S") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, onei},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "T") {
            std::complex<double> c = (1.0 + onei) / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "I") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "Rzy") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> c_i = onei / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c_i, +c},
                {+c, +c_i},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "rU1") {
            std::complex<double> a = std::cos(parameter);
            std::complex<double> b = std::sin(parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {+b, +a},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }

    } else {
        if (type == "A") {
            std::complex<double> c = std::cos(parameter);
            std::complex<double> s = std::sin(parameter);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, c, s, 0.0},
                {0.0, s, -c, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, false));
        }
        if ((type == "cX") or (type == "CNOT")) {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if ((type == "acX") or (type == "aCNOT")) {
            std::complex<double> gate[4][4]{
                {0.0, 1.0, 0.0, 0.0},
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cY") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, -onei},
                {0.0, 0.0, +onei, 0.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cZ") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cR") {
            std::complex<double> tmp = onei * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, c},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "cV") {
            std::complex<double> a = onei * 0.5 + 0.5;
            std::complex<double> b = -onei * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +a, +b},
                {0.0, 0.0, +b, +a},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "cRz") {
            std::complex<double> tmp_a = -onei * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = onei * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, a, 0.0},
                {0.0, 0.0, 0.0, b},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
        if (type == "SWAP") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return Gate(type, target, control, gate);
        }
        if (type == "rU2") {
            std::complex<double> a = std::cos(parameter);
            std::complex<double> b = std::sin(parameter);
            std::complex<double> gate[4][4]{
                {+a, -b, 0.0, 0.0},
                {+b, +a, 0.0, 0.0},
                {0.0, 0.0, +a, -b},
                {0.0, 0.0, +b, +a},
            };
            return Gate(type, target, control, gate, std::make_pair(parameter, true));
        }
    }
    // If you reach this section then the gate type is not implemented or it is invalid.
    // So we throw an exception that propagates to Python and return the identity
    std::string msg = fmt::format("make_gate()\ntype = {} is not a valid quantum gate type", type);
    throw std::invalid_argument(msg);
    std::complex<double> gate[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    return Gate(type, target, control, gate);
}

Gate make_control_gate(size_t control, Gate& U) {
    // using namespace std::complex_literals;
    std::string type = "cU";
    size_t target = U.target();
    if (target == control) {
        std::string msg = fmt::format("Cannot create Control-U where targer == control !");
        throw std::invalid_argument(msg);
    }
    const auto& mat = U.matrix();
    std::complex<double> gate[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, mat[0][0], mat[0][1]},
        {0.0, 0.0, mat[1][0], mat[1][1]},
    };
    auto parameter_info =
        U.has_parameter() ? std::make_pair(U.parameter().value(), U.minus_parameter_on_adjoint())
                          : std::optional<std::pair<double, bool>>();
    return Gate(type, target, control, gate, parameter_info);
}