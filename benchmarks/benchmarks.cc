#include <iostream>
#include <fstream> // std::filebuf

#include "hayai/hayai.hpp"
//#include "hayai/hayai_main.hpp"

#include "quantum_basis.h"
#include "quantum_circuit.h"
#include "quantum_computer.h"
#include "quantum_gate.h"

QuantumBasis qb;
size_t acc;
constexpr int nqbits = 10;
QuantumComputer qc(nqbits);
QuantumCircuit qcirc;
QuantumGate qg = make_gate("X", 5, 5);

void prepare_circ(QuantumCircuit& qcirc, size_t nqbits) {
    for (size_t i = 0; i < nqbits; i++) {
        qcirc.add_gate(make_gate("X", i, i));
        //        qcirc.add_gate(make_gate("Y", i, i));
        //        qcirc.add_gate(make_gate("Z", i, i));
    }
}

int main(int argc, char* argv[]) {
    for (int i = 0; i < nqbits; i++) {
        qc.apply_gate(make_gate("H", i, i));
    }
    prepare_circ(qcirc, 10);
    hayai::ConsoleOutputter consoleOutputter;
    hayai::Benchmarker::AddOutputter(consoleOutputter);
    hayai::Benchmarker::RunAllTests();
    std::cout << acc;
    return 0;

    //    std::filebuf fb;
    //    fb.open("bench.json", std::ios::out);
    //    std::ostream json(&fb);

    //    // Set up the main runner.
    //    hayai::MainRunner runner;

    //    // Parse the arguments.
    //    int result = runner.ParseArgs(argc, argv);
    //    if (result)
    //        return result;

    //    //    hayai::ConsoleOutputter consoleOutputter;
    //    //    hayai::JsonOutputter JSONOutputter(json);

    //    //    hayai::Benchmarker::AddOutputter(consoleOutputter);
    //    //    hayai::Benchmarker::AddOutputter(JSONOutputter);
    //    //    hayai::Benchmarker::RunAllTests();

    //    // Execute based on the selected mode.
    //    auto return_value = runner.Run();
    //    fb.close();
    //    return return_value;
}

BENCHMARK(QuantumBasis, set_bit, 10, 100000) {
    for (int i = 0; i < 64; ++i) {
        acc += qb.set_bit(i, true);
        acc += qb.set_bit(i, false);
    }
}

BENCHMARK(QuantumBasis, set_bit2, 10, 100000) {
    for (int i = 0; i < 64; ++i) {
        qb.set_bit2(i, true);
        qb.set_bit2(i, false);
    }
}

BENCHMARK(QuantumBasis, set_bit3, 10, 100000) {
    for (int i = 0; i < 64; ++i) {
        qb.set_bit3(i, true);
        qb.set_bit3(i, false);
    }
}

BENCHMARK(QuantumComputer, apply_gate, 100, 1000) { qc.apply_gate(qg); }
//BENCHMARK(QuantumComputer, apply_1qubit_gate, 100, 1000) { qc.apply_1qubit_gate(qg); }

 BENCHMARK(QuantumComputer, apply_circuit, 100, 1000) { qc.apply_circuit(qcirc); }

// BENCHMARK(QuantumComputer, apply_circuit, 100, 1000) { qc.apply_circuit(qcirc); }

// BENCHMARK(Determinant128, get_bit 10, 100)
//{
//    det32_a.get_bit(19);
//}

// BENCHMARK_P(Determinant128, get_alfa_bit, 10, 1000000, (std::size_t n)) {
//    acc1 += det128_a.get_alfa_bit(n);
//}
// BENCHMARK_P_INSTANCE(Determinant128, get_alfa_bit, (2));

// BENCHMARK_P(Determinant128, get_bit, 10, 1000000, (std::size_t n)) { det128_a.get_bit(n); }
// BENCHMARK_P_INSTANCE(Determinant128, get_bit, (2));

// BENCHMARK_P(Determinant128, get_alfa_bit, 10, 1000000, (std::size_t n)) { acc1 +=
// det128_a.get_alfa_bit(n); } BENCHMARK_P_INSTANCE(Determinant128, get_alfa_bit, (2));

// BENCHMARK_P(Determinant128, get_beta_bit, 10, 1000000, (std::size_t n)) {
// det128_a.get_beta_bit(n); } BENCHMARK_P_INSTANCE(Determinant128, get_beta_bit, (2));

// BENCHMARK(Determinant128, compare, 10, 1000000)
//{ det128_a == det128_b; }

// BENCHMARK(Determinant128, allocate, 10, 100000)
//{ Determinant<128> d; }

// Determinant<1024> det1024_a;
// Determinant<1024> det1024_b = det1024_a;

// BENCHMARK(Determinant1024, compare, 10, 1000000)
//{ det1024_a == det1024_b; }

// BENCHMARK(Determinant1024, allocate, 10, 100000)
//{ Determinant<1024> d; }

// BENCHMARK(UI64Determinant, count, 10, 1000) {
//    det_test.count_alfa();
//    det_test.count_beta();
//}

// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (2));

// BENCHMARK_P(UI64Determinant, sign_a, 10, 10000, (std::size_t n)) { det_test.slater_sign_a(n); }

// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (2));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (16));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (32));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (63));

// UI64Determinant make_det_from_string(std::string s_a, std::string s_b) {
//    UI64Determinant d;
//    if (s_a.size() == s_b.size()) {
//        for (std::string::size_type i = 0; i < s_a.size(); ++i) {
//            d.set_alfa_bit(i, s_a[i] == '0' ? 0 : 1);
//            d.set_beta_bit(i, s_b[i] == '0' ? 0 : 1);
//        }
//    } else {
//        std::cout << "\n\n  Function make_det_from_string called with strings of different size";
//        exit(1);
//    }
//    return d;
//}

// UI64Determinant det_test =
//    make_det_from_string("1001100000000000000000000000000000000000000000000000000000010000",
//                         "0001000000000000001000000000000000000000000000000000000000000001");

// BENCHMARK(UI64Determinant, count, 10, 1000) {
//    det_test.count_alfa();
//    det_test.count_beta();
//}

// BENCHMARK_P(UI64Determinant, sign_a, 10, 10000, (std::size_t n)) { det_test.slater_sign_a(n); }

// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (2));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (16));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (32));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_a, (63));

// BENCHMARK_P(UI64Determinant, sign_aa, 10, 10000, (std::size_t m, std::size_t n)) {
//    det_test.slater_sign_aa(m, n);
//}

// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aa, (1, 2));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aa, (8, 16));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aa, (16, 32));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aa, (32, 63));

// BENCHMARK_P(UI64Determinant, sign_aaaa, 10, 10000, (int i, int j, int a, int b)) {
//    det_test.slater_sign_aaaa(i, j, a, b);
//}

// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aaaa, (1, 4, 32, 63));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aaaa, (1, 4, 63, 32));
// BENCHMARK_P_INSTANCE(UI64Determinant, sign_aaaa, (63, 32, 1, 4));
