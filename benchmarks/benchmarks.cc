#include <iostream>
#include <fstream> // std::filebuf

//#include "hayai/hayai.hpp"
//#include "hayai/hayai_main.hpp"

#include "quantum_basis.h"
#include "quantum_circuit.h"
#include "quantum_computer.h"
#include "quantum_gate.h"

QuantumBasis qb;

QuantumComputer qc_4(4);
QuantumComputer qc_8(8);
QuantumComputer qc_16(16);
QuantumComputer qc_18(18);

QuantumCircuit qcirc_4;
QuantumCircuit qcirc_8;
QuantumCircuit qcirc_16;
QuantumCircuit qcirc_18;

QuantumGate qg = make_gate("X", 7, 7);

std::vector<QuantumComputer> computers;

void prepare_circ(QuantumCircuit& qcirc, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        qcirc.add_gate(make_gate("X", i, i));
    }
}

// prepare_circ(qcirc_8, 0, 8);
// prepare_circ(qcirc_16, 0, 16);
// prepare_circ(qcirc_18, 0, 18);

#include <string>
#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp>

using namespace std;

TEST_CASE("QuantumComputer", "[benchmark]") {
    prepare_circ(qcirc_4, 0, 4);
    prepare_circ(qcirc_8, 0, 8);
    prepare_circ(qcirc_16, 0, 16);
    prepare_circ(qcirc_18, 0, 18);

    BENCHMARK("qc_4_apply_circuit") { qc_4.apply_circuit(qcirc_4); }

    BENCHMARK("qc_8_apply_circuit") { qc_8.apply_circuit(qcirc_8); }

    BENCHMARK("qc_16_apply_circuit") { qc_16.apply_circuit(qcirc_16); }

    BENCHMARK("qc_18_apply_circuit") { qc_18.apply_circuit(qcirc_18); }
}

// int main(int argc, char* argv[]) {
//    prepare_circ(qcirc_8, 0, 8);
//    prepare_circ(qcirc_16, 0, 16);
//    prepare_circ(qcirc_18, 0, 18);

//    // Set up the main runner.
//    hayai::MainRunner runner;

//    // Parse the arguments.
//    int result = runner.ParseArgs(argc, argv);
//    if (result)
//        return result;

//    // Execute based on the selected mode.
//    return runner.Run();
//    //    hayai::ConsoleOutputter consoleOutputter;
//    //    hayai::Benchmarker::AddOutputter(consoleOutputter);
//    //    hayai::Benchmarker::RunAllTests();
//    return 0;
//}

// BENCHMARK(QuantumBasis, set_bit, 10, 100000) {
//    for (int i = 0; i < 64; ++i) {
//        qb.set_bit(i, true);
//        qb.set_bit(i, false);
//    }
//}

// BENCHMARK(QuantumComputer, qc_8_apply_gate, 100, 1) { qc_8.apply_circuit(qcirc_8); }
// BENCHMARK(QuantumComputer, qc_16_apply_gate, 100, 1) { qc_16.apply_circuit(qcirc_16); }
// BENCHMARK(QuantumComputer, qc_18_apply_gate, 100, 1) { qc_18.apply_circuit(qcirc_18); }

// BENCHMARK(QuantumComputer, qc_8_apply_gate_fast, 100, 1) { qc_8.apply_circuit_fast(qcirc_8); }
// BENCHMARK(QuantumComputer, qc_16_apply_gate_fast, 100, 1) { qc_16.apply_circuit_fast(qcirc_16); }
// BENCHMARK(QuantumComputer, qc_18_apply_gate_fast, 100, 1) { qc_18.apply_circuit_fast(qcirc_18); }
