#include <stdexcept>

#include "fmt/format.h"

#include "quantum_gate.h"

QuantumGate make_gate(std::string type, size_t target, size_t control, double parameter) {
    using namespace std::complex_literals;
    if (target == control) {
        if (type == "X") {
            std::complex<double> gate[4][4]{
                {0.0, 1.0},
                {1.0, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Y") {
            std::complex<double> gate[4][4]{
                {0.0, -1.0i},
                {+1.0i, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Z") {
            std::complex<double> gate[4][4]{
                {+1.0, 0.0},
                {0.0, -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "H") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c, +c},
                {+c, -c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "R") {
	    std::complex<double> tmp = 1.0i * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Rx") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = 1.0i * std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {-b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Ry") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {+a, -b},
                {+b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Rz") {
            std::complex<double> tmp_a = -1.0i * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = 1.0i * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {a, 0.0},
                {0.0, b},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "V") {
            std::complex<double> a = 1.0i * 0.5 + 0.5;
            std::complex<double> b = -1.0i * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {+a, +b},
                {+b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "S") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0i},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "T") {
            std::complex<double> c = (1.0 + 1.0i) / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "I") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0},
                {0.0, 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "Rzy") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> c_i = 1.0i / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {+c_i, +c},
                {+c, +c_i},
            };
            return QuantumGate(type, target, control, gate);
        }

    } else {
        if ((type == "cX") or (type == "CNOT")) {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
                {0.0, 0.0, 1.0, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cY") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, -1.0i},
                {0.0, 0.0, +1.0i, 0.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cZ") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, -1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cR") {
            std::complex<double> tmp = 1.0i * parameter;
            std::complex<double> c = std::exp(tmp);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 0.0, 0.0, c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cRx") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = 1.0i * std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +a, -b},
                {0.0, 0.0, -b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cRy") {
            std::complex<double> a = std::cos(0.5 * parameter);
            std::complex<double> b = std::sin(0.5 * parameter);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +a, -b},
                {0.0, 0.0, +b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cRz") {
            std::complex<double> tmp_a = -1.0i * 0.5 * parameter;
            std::complex<double> a = std::exp(tmp_a);
            std::complex<double> tmp_b = 1.0i * 0.5 * parameter;
            std::complex<double> b = std::exp(tmp_b);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, a, 0.0},
                {0.0, 0.0, 0.0, b},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cH") {
            std::complex<double> c = 1.0 / std::sqrt(2.0);
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +c, +c},
                {0.0, 0.0, +c, -c},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "cV") {
            std::complex<double> a = 1.0i * 0.5 + 0.5;
            std::complex<double> b = -1.0i * 0.5 + 0.5;
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, +a, +b},
                {0.0, 0.0, +b, +a},
            };
            return QuantumGate(type, target, control, gate);
        }
        if (type == "SWAP") {
            std::complex<double> gate[4][4]{
                {1.0, 0.0, 0.0, 0.0},
                {0.0, 0.0, 1.0, 0.0},
                {0.0, 1.0, 0.0, 0.0},
                {0.0, 0.0, 0.0, 1.0},
            };
            return QuantumGate(type, target, control, gate);
        }
    }
    // If you reach this section then the gate type is not implemented or it is invalid.
    // So we throw an exception that propagates to Python and return the identity
    std::string msg =
        fmt::format("make_quantum_gate()\ntype = {} is not a valid quantum gate type", type);
    throw std::invalid_argument(msg);
    std::complex<double> gate[4][4]{
        {1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 1.0},
    };
    return QuantumGate(type, target, control, gate);
}

QuantumGate make_control_gate(size_t control, QuantumGate& U) {
    using namespace std::complex_literals;
    std::string type = "cU";
    size_t target = U.target();
    if (target == control) {
        std::string msg =
            fmt::format("Cannot create Control-U where targer == control !");
        throw std::invalid_argument(msg);
    }
    std::complex<double> a = U.gate()[0][0];
    std::complex<double> b = U.gate()[0][1];
    std::complex<double> c = U.gate()[1][0];
    std::complex<double> d = U.gate()[1][1];
    std::complex<double> gate[4][4]{
            {1.0, 0.0, 0.0, 0.0},
            {0.0, 1.0, 0.0, 0.0},
            {0.0, 0.0, a, b},
            {0.0, 0.0, c, d},
        };
    return QuantumGate(type, target, control, gate);
}
