#define CATCH_CONFIG_MAIN
#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch.hpp>

#include "qubit_basis.h"
#include "circuit.h"
#include "computer.h"
#include "qubit_op_pool.h"
#include "qubit_operator.h"
#include "gate.h"

QubitBasis qb;

Circuit qcirc_18;

Circuit qcirc2_18;

Circuit qcirc_2qb_18;

Circuit qcirc2_2qb_18;

std::vector<Computer> computers;

void prepare_circ(Circuit& qcirc, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        qcirc.add_gate(make_gate("H", i, i));
    }
}

void prepare_circ2(Circuit& qcirc, size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
        qcirc.add_gate(make_gate("X", i, i));
        qcirc.add_gate(make_gate("Y", i, i));
        qcirc.add_gate(make_gate("Z", i, i));
    }
}

void prepare_2q_circ(Circuit& qcirc, size_t start, size_t end) {
    for (size_t i = start; i < end - 1; i++) {
        qcirc.add_gate(make_gate("cX", i, i + 1));
        qcirc.add_gate(make_gate("cX", i + 1, i));
    }
}

void prepare_2q_circ2(Circuit& qcirc, size_t start, size_t end) {
    for (size_t i = start; i < end - 1; i++) {
        qcirc.add_gate(make_gate("cX", i, i + 1));
        qcirc.add_gate(make_gate("cX", i + 1, i));
        qcirc.add_gate(make_gate("cY", i, i + 1));
        qcirc.add_gate(make_gate("cY", i + 1, i));
        qcirc.add_gate(make_gate("cZ", i, i + 1));
        qcirc.add_gate(make_gate("cZ", i + 1, i));
        qcirc.add_gate(make_gate("cR", i, i + 1, 3.14159 / (i + 1.0)));
        qcirc.add_gate(make_gate("cR", i + 1, i, 2.17284 / (i + 1.0)));
    }
}

TEST_CASE("Computer_1qubit_gate_18qubits", "[benchmark]") {

    prepare_circ(qcirc_18, 0, 18);
    prepare_circ2(qcirc2_18, 0, 18);

    Computer qc1_18(18);
    BENCHMARK("qc_18_apply_circuit_safe") { qc1_18.apply_circuit_safe(qcirc_18); };

    // Computer qc2_18(18);
    // BENCHMARK("qc_18_apply_circuit_fast") { qc2_18.apply_circuit_fast(qcirc_18); };

    Computer qc3_18(18);
    BENCHMARK("qc_18_apply_circuit") { qc3_18.apply_circuit(qcirc_18); };

    Computer qc4_18(18);
    BENCHMARK("qc_18_apply_circuit2_safe") { qc4_18.apply_circuit_safe(qcirc2_18); };

    // Computer qc5_18(18);
    // BENCHMARK("qc_18_apply_circuit2_fast") { qc5_18.apply_circuit_fast(qcirc2_18); };

    Computer qc6_18(18);
    BENCHMARK("qc_18_apply_circuit2") { qc6_18.apply_circuit(qcirc2_18); };
}

TEST_CASE("Computer_2qubit_gate_18qubits", "[benchmark]") {

    prepare_2q_circ(qcirc_2qb_18, 0, 18);
    prepare_2q_circ2(qcirc2_2qb_18, 0, 18);

    // For qcirc_2qb_18 (many cX gates)
    Computer qc1_18(18);
    BENCHMARK("qc_18_apply_2qb_circuit_safe") { qc1_18.apply_circuit_safe(qcirc_2qb_18); };

    Computer qc3_18(18);
    BENCHMARK("qc_18_apply_2qb_circuit") { qc3_18.apply_circuit(qcirc_2qb_18); };

    // For qcirc2_2qb_18 (variety of gates)
    Computer qc4_18(18);
    BENCHMARK("qc_18_apply_2qb_circuit2_safe") { qc4_18.apply_circuit_safe(qcirc2_2qb_18); };

    Computer qc6_18(18);
    BENCHMARK("qc_18_apply_2qb_circuit2") { qc6_18.apply_circuit(qcirc2_2qb_18); };
}
