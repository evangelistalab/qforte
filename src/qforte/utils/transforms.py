"""
transforms.py
=================================================
A module for operator transform functions,
from second quantization -> qubit representation.
"""

import qforte
import numpy as np
import copy


def fermop_to_sq_excitation(fermop):
    # fermop => list of tuples [ ((idx, action), (idx, action), ...), coeff)  ]
    sq_excitations = []
    for term in fermop:
        coeff = term[1]
        tup = term[0]

        num_ops = len(tup)
        if int(num_ops % 2) != 0:
            raise ValueError(
                "Second quantized term must have an even number of operators."
            )

        an_count = 0
        cr_count = 0
        term_lst = []
        for i, an_cr in enumerate(tup):
            # print('i: ', i, ' int(num_ops/2): ', int(num_ops/2), ' cr_count: ', cr_count )
            # check to make sure tuple is already normal ordered
            if i < int(num_ops / 2) and an_count > 0:
                # print('here1')
                raise ValueError("Term is not normal ordered!")

            if i >= int(num_ops / 2) and an_count > int(num_ops / 2):
                # print('here2')
                raise ValueError("Term is not normal ordered!")

            term_lst.append(an_cr[0])

            if an_cr[1]:
                cr_count += 1
            else:
                an_count += 1

        if an_count != cr_count:
            raise ValueError("Term is not particle number conserving!")

        sq_excitations.append([tuple(term_lst), coeff])

    return sq_excitations


# TODO: Rename organizer to operator (Nick)
def organizer_to_circuit(op_organizer):
    """Builds a Circuit from a operator orgainizer.

    Parameters
    ----------
    op_organizer : list
        An object to organize what the coefficient and Pauli operators in terms
        of the QubitOperator will be.

        The orginzer is of the form
        [[coeff_a, [ ("X", i), ("Z", j),  ("Y", k), ...  ] ], [...] ...]
        where X, Y, Z are strings that indicate Pauli opterators;
        i, j, k index qubits and coeff_a indicates the coefficient for the ath
        term in the QubitOperator.
    """
    operator = qforte.QubitOperator()
    for coeff, word in op_organizer:
        circ = qforte.Circuit()
        for letter in word:
            circ.add(qforte.gate(letter[0], letter[1], letter[1]))

        operator.add(coeff, circ)

    return operator


# TODO: Rename operator to organizer (Nick)
def circuit_to_organizer(operator):
    """Builds a operator orgainizer from a Circuit.

    Parameters
    ----------
    operator : QubitOperator
        The QubitOperator object to converted to organizer form

        The orginzer is of the form
        [[coeff_a, [ ("X", i), ("Z", j),  ("Y", k), ...  ] ], [...] ...]
        where X, Y, Z are strings that indicate Pauli opterators;
        i, j, k index qubits and coeff_a indicates the coefficient for the ath
        term in the QubitOperator.
    """
    op_organizer = []
    for term in operator.terms():
        term_organizer = []
        gates_in_term = []
        term_organizer.append(term[0])
        for gate in term[1].gates():
            gates_in_term.append((gate.gate_id(), gate.target()))

        term_organizer.append(gates_in_term)
        op_organizer.append(term_organizer)

    return op_organizer


def get_ucc_jw_organizer(sq_excitations, already_anti_herm=False):
    # TODO: rename function to be general i.e. "get_jw_organizer" (Nick)
    # TODO: write test case for the "already_anit_herm=False" case (Nick)

    T_organizer = []

    if already_anti_herm:
        for sq_term in sq_excitations:
            T_organizer.append(get_single_term_jw_organizer(sq_term))

    else:
        # print('\nsq_excitation: ', sq_excitations)
        for sq_op, amp in sq_excitations:
            # sq_op, amp => (p,q), t_p^q
            # print('sq_term: ', sq_term)

            sq_term = [sq_op, amp]
            sq_term_dag = [sq_op[::-1], -1.0 * amp]

            T_organizer.append(get_single_term_jw_organizer(sq_term))
            T_organizer.append(get_single_term_jw_organizer(sq_term_dag))

    T_organizer = combine_like_terms(T_organizer)

    return T_organizer


def get_jw_organizer(sq_excitations, combine=True):
    organizer = []
    for sq_term in sq_excitations:
        organizer.append(get_single_term_jw_organizer(sq_term))

    if combine:
        organizer = combine_like_terms(organizer)

    return organizer


def combine_like_terms(op_organizer):
    # TODO (opt): A very slow implementation, could absolutely be improved
    term_coeff_dict = {}
    combined_op_organizer = []
    threshold = 1.0e-10
    for jw_term in op_organizer:
        for coeff, word in jw_term:
            temp_tup = tuple(word)
            if temp_tup not in term_coeff_dict.keys():
                term_coeff_dict[temp_tup] = coeff
            else:
                term_coeff_dict[temp_tup] += coeff

    for word in term_coeff_dict:
        if np.abs(term_coeff_dict[word]) > threshold:
            combined_op_organizer.append([term_coeff_dict[word], list(word)])

    return combined_op_organizer


def get_single_term_jw_organizer(sq_term):
    # sq_term => [(p,q), t_p^q]
    n_creators = int(len(sq_term[0]) / 2)
    sq_ops = sq_term[0]
    sq_coeff = sq_term[1]

    #                 'coef'                  'word'
    # organizer => [[coeff_0, [ ("X", i), ("X", j),  ("X", k), ...  ] ], [...] ...]
    op_organizer = []

    if len(sq_ops) == 0:
        op_organizer.append([sq_coeff, []])
        return op_organizer

    ### For right side operator (a single anihilator or creator)
    for i, op_idx in enumerate(sq_ops):
        r_op_Xorganizer = []
        r_op_Yorganizer = []
        for j in range(op_idx):
            r_op_Xorganizer.append(("Z", j))
            r_op_Yorganizer.append(("Z", j))

        r_op_Xorganizer.append(("X", op_idx))
        r_op_Yorganizer.append(("Y", op_idx))

        rX_coeff = 0.5

        if i < n_creators:
            rY_coeff = -0.5j
        else:
            rY_coeff = 0.5j

        r_X_term = [rX_coeff, r_op_Xorganizer]
        r_Y_term = [rY_coeff, r_op_Yorganizer]

        op_organizer = join_lr_organizers(op_organizer, r_X_term, r_Y_term)

    for i in range(len(op_organizer)):
        op_organizer[i][0] *= sq_coeff

    return op_organizer


def join_organizers(L_op_org, R_op_org):
    combined_op_org = []
    for Lcoeff, Lword in L_op_org:
        for Rcoeff, Rword in R_op_org:
            comb_coeff = Lcoeff * Rcoeff
            comb_word = Lword + Rword
            combined_op_org.append([comb_coeff, comb_word])

    return combine_like_terms([pauli_condense(combined_op_org)])


# def get_org_idxs(org):
#     pass
#
# def get_word_idxs(word):
#     """Gets a list of indexes corresponing to a word
#     """
#     indices = []
#     pass


# works similarly to join_orgainzers() but assumes H and Am as operators
# esentially builds a commutator HAm - AmH
def join_H_Am_organizers(H_org, Am_org):
    # word =>[ ("X", i), ("X", j),  ("X", k), ...  ]

    # find what indices are present in Am_org
    # Am_indices = []
    # for Rcoeff, Rword in Am_org:
    #     for Rletter in Rword:
    #         idx = Rword
    #     if

    combined_op_org = []
    for Lcoeff, Lword in H_org:
        for Rcoeff, Rword in Am_org:
            comb_coeff = Lcoeff * Rcoeff
            comb_word = Lword + Rword
            combined_op_org.append([comb_coeff, comb_word])
            comb_word2 = Rword + Lword
            combined_op_org.append([-comb_coeff, comb_word2])

    return combine_like_terms([pauli_condense(combined_op_org)])


def join_lr_organizers(current_op_org, r_op_Xterm, r_op_Yterm):
    if not current_op_org:
        combined_op_org = [r_op_Xterm, r_op_Yterm]

    else:
        combined_op_org = []
        for coeff, word in current_op_org:
            combX_coeff = coeff * r_op_Xterm[0]
            combX_word = word + r_op_Xterm[1]
            combined_op_org.append([combX_coeff, combX_word])

            combY_coeff = coeff * r_op_Yterm[0]
            combY_word = word + r_op_Yterm[1]
            combined_op_org.append([combY_coeff, combY_word])

    return pauli_condense(combined_op_org)
    """return object : [
                       [c_i, [ ('Z', qi1), ('X', qi2), ... ] ],
                       [c_j, [ ('Y', qj1), ... ] ]
                       ] """


def pauli_condense(pauli_op):
    condensed_op = []
    contractions = {
        ("X", "Y"): (1.0j, "Z"),
        ("X", "Z"): (-1.0j, "Y"),
        ("Y", "X"): (-1.0j, "Z"),
        ("Y", "Z"): (1.0j, "X"),
        ("Z", "X"): (1.0j, "Y"),
        ("Z", "Y"): (-1.0j, "X"),
        ("X", "X"): (1.0, "I"),
        ("Y", "Y"): (1.0, "I"),
        ("Z", "Z"): (1.0, "I"),
        ("I", "X"): (1.0, "X"),
        ("I", "Y"): (1.0, "Y"),
        ("I", "Z"): (1.0, "Z"),
    }

    for current_coeff, current_word in pauli_op:
        # sort the pauli ops by increasing qubit
        word = sorted(current_word, key=lambda factor: factor[1])
        condensed_word = [current_coeff, []]
        # loop over the letters and pairwise compare then to simplify
        # (coeff, [ ('sigma', idx), ('sigma', idx), ... ] )
        # will modify coeff and delete/replace tuples from the list

        idx1 = 0

        while idx1 < len(word):
            current_qubit = word[idx1][1]
            temp_list = []
            condensed_temp_list = []
            idx2 = copy.copy(idx1)

            while idx2 < len(word) and word[idx2][1] == current_qubit:
                temp_list.append((word[idx2][0], word[idx2][1]))
                idx2 += 1

            if len(temp_list) == 1:
                condensed_word[1].append(temp_list[0])
                idx1 += len(temp_list)

            else:
                idx3 = 0
                while idx3 < len(temp_list) - 1:
                    condensed_word[0] *= contractions[
                        (temp_list[idx3][0], temp_list[idx3 + 1][0])
                    ][0]
                    new_letter = contractions[
                        (temp_list[idx3][0], temp_list[idx3 + 1][0])
                    ][1]
                    condensed_temp_list.append((new_letter, current_qubit))
                    idx3 += 1

                for letter in condensed_temp_list[0][0]:
                    if letter != "I":
                        condensed_word[1].append((letter, current_qubit))

                idx1 += len(temp_list)

        condensed_op.append(condensed_word)

    return condensed_op
