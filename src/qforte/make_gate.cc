#include <stdexcept>

#include "fmt/format.h"

#include "quantum_gate.h"

QuantumGate make_gate(std::string type, size_t target, size_t control, double parameter, bool mirror) {
    using namespace std::complex_literals;
    if (target == control) {
        if(mirror) {
            if (type == "X") {
                type = "H";
            }
            else if (type == "Y") {
                type = "Rzy";
            }
            else if (type == "Z") {
                type = "I";
            } else {
                std::string msg =
                    fmt::format("Mirror gate\ntype = {} can only be of type X, Y or Z,", type);
                throw std::invalid_argument(msg);
            }
        }

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
            std::complex<double> a = std::exp(-1.0i * 0.5 * parameter);
            std::complex<double> b = std::exp(1.0i * 0.5 * parameter);
            std::complex<double> gate[4][4]{
                {[0] = a},
                {[1] = b},
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
                {+c, +c},
                {+c_i, -c_i},
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
