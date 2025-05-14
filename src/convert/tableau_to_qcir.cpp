/****************************************************************************
  PackageName  [ tableau ]
  Synopsis     [ Define pauli rotation class ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./tableau_to_qcir.hpp"

#include <cassert>
#include <gsl/narrow>
#include <random>
#include <stack>
#include <tl/adjacent.hpp>
#include <tl/enumerate.hpp>
#include <tl/to.hpp>

#include "qcir/basic_gate_type.hpp"
#include "qcir/qcir.hpp"
#include "util/graph/digraph.hpp"
#include "util/graph/minimum_spanning_arborescence.hpp"
#include "util/phase.hpp"
#include "util/util.hpp"

extern bool stop_requested();

namespace qsyn::experimental {

namespace {

void add_clifford_gate(qcir::QCir& qcir, CliffordOperator const& op) {
    using COT                  = CliffordOperatorType;
    auto const& [type, qubits] = op;

    switch (type) {
        case COT::h:
            qcir.append(qcir::HGate(), {qubits[0]});
            break;
        case COT::s:
            qcir.append(qcir::SGate(), {qubits[0]});
            break;
        case COT::cx:
            qcir.append(qcir::CXGate(), {qubits[0], qubits[1]});
            break;
        case COT::sdg:
            qcir.append(qcir::SdgGate(), {qubits[0]});
            break;
        case COT::v:
            qcir.append(qcir::SXGate(), {qubits[0]});
            break;
        case COT::vdg:
            qcir.append(qcir::SXdgGate(), {qubits[0]});
            break;
        case COT::x:
            qcir.append(qcir::XGate(), {qubits[0]});
            break;
        case COT::y:
            qcir.append(qcir::YGate(), {qubits[0]});
            break;
        case COT::z:
            qcir.append(qcir::ZGate(), {qubits[0]});
            break;
        case COT::cz:
            qcir.append(qcir::CZGate(), {qubits[0], qubits[1]});
            break;
        case COT::swap:
            qcir.append(qcir::SwapGate(), {qubits[0], qubits[1]});
            break;
        case COT::ecr:
            qcir.append(qcir::ECRGate(), {qubits[0], qubits[1]});
            break;
    }
}
}  // namespace

/**
 * @brief convert a stabilizer tableau to a QCir.
 *
 * @param clifford - pass by value on purpose
 * @return std::optional<qcir::QCir>
 */
std::optional<qcir::QCir> to_qcir(StabilizerTableau const& clifford, StabilizerTableauSynthesisStrategy const& strategy) {
    qcir::QCir qcir{clifford.n_qubits()};
    for (auto const& op : extract_clifford_operators(clifford, strategy)) {
        if (stop_requested()) {
            return std::nullopt;
        }
        add_clifford_gate(qcir, op);
    }

    return qcir;
}

std::optional<qcir::QCir> NaivePauliRotationsSynthesisStrategy::synthesize(std::vector<PauliRotation> const& rotations) const {
    if (rotations.empty()) {
        return qcir::QCir{0};
    }

    auto qcir = qcir::QCir{rotations.front().n_qubits()};

    for (auto const& rotation : rotations) {
        auto [ops, qubit] = extract_clifford_operators(rotation);

        for (auto const& op : ops) {
            add_clifford_gate(qcir, op);
        }

        qcir.append(qcir::PZGate(rotation.phase()), {qubit});

        adjoint_inplace(ops);

        for (auto const& op : ops) {
            add_clifford_gate(qcir, op);
        }
    }

    return qcir;
}

std::optional<qcir::QCir> TParPauliRotationsSynthesisStrategy::synthesize(std::vector<PauliRotation> const& /* rotations */) const {
    spdlog::error("TPar Synthesis Strategy is not implemented yet!!");
    return std::nullopt;
}

namespace {
/**
 * @brief select a row consisting completely of 1s to be the target row.
 *
 * @param rotations
 * @param rotation_filter
 * @param pivot
 * @return size_t
 */
std::vector<size_t>
get_control_rows(
    std::vector<PauliRotation> const& rotations,
    std::vector<size_t> const& rotation_filter,
    size_t pivot) {
    auto const num_qubits = rotations.front().n_qubits();
    auto control_rows     = std::vector<size_t>{};
    for (auto i : std::views::iota(0ul, num_qubits)) {
        if (i == pivot) continue;

        if (std::ranges::all_of(
                rotation_filter,
                [&](auto x) {
                    return rotations[x].pauli_product().is_z_set(i);
                })) {
            control_rows.push_back(i);
        }
    }

    return control_rows;
}

void apply_cxs(
    std::vector<size_t> ctrls,
    size_t targ,
    GraySynthPauliRotationsSynthesisStrategy::Mode mode,
    std::vector<PauliRotation>& rotations,
    qcir::QCir& qcir,
    StabilizerTableau& final_clifford,
    std::unordered_set<std::size_t> const& frozen_rotations,
    std::size_t num_rotations,
    std::vector<std::size_t> const& random_order) {
    using Mode = GraySynthPauliRotationsSynthesisStrategy::Mode;

    auto const apply_cx = [&](size_t ctrl, size_t targ) {
        for (auto col_id : std::views::iota(0ul, num_rotations)) {
            if (!frozen_rotations.contains(col_id)) {
                rotations[col_id].cx(ctrl, targ);
            }
        }
        qcir.append(qcir::CXGate(), {ctrl, targ});
        final_clifford.prepend_cx(ctrl, targ);
    };

    switch (mode) {
        case Mode::star:
            for (auto ctrl : ctrls) {
                apply_cx(ctrl, targ);
            }
            break;
        case Mode::staircase:
            // sort the controls according to the random_order
            std::ranges::sort(ctrls, [&](auto const& x, auto const& y) {
                return random_order[x] < random_order[y];
            });
            for (auto&& [c, t] : ctrls | tl::views::pairwise) {
                apply_cx(c, t);
            }
            if (!ctrls.empty()) {
                apply_cx(ctrls.back(), targ);
            }
            break;
    }
}

/**
 * @brief select a row with the most or least number of 1.
 *
 * @param rotations
 * @param rotation_filter
 * @param qubit_filter
 * @return size_t
 */
size_t
get_cofactor_row(std::vector<PauliRotation> const& rotations, std::vector<size_t> const& rotation_filter, std::vector<size_t> const& qubit_filter) {
    auto counts = std::vector<std::size_t>(qubit_filter.size(), 0);
    for (auto col_id : rotation_filter) {
        for (auto&& [idx, qubit] : tl::views::enumerate(qubit_filter)) {
            if (rotations[col_id].pauli_product().is_z_set(qubit)) {
                counts[idx]++;
            }
        }
    }

    auto const [min_it, max_it] = std::ranges::minmax_element(counts);
    auto const most_ones        = std::distance(counts.begin(), max_it);
    auto const most_zeros       = std::distance(counts.begin(), min_it);

    if (counts[most_ones] >= rotation_filter.size() - counts[most_zeros]) {
        return qubit_filter[most_ones];
    } else {
        return qubit_filter[most_zeros];
    }
}

/**
 * @brief filter out a number from a vector.
 *
 * @param vec
 * @param num
 * @return std::vector<std::size_t>
 */
std::vector<std::size_t>
filter_out_number(
    std::vector<std::size_t> const& vec,
    std::size_t num) {
    return vec |
           std::views::filter([&](auto const& x) { return x != num; }) |
           tl::to<std::vector>();
}

}  // namespace

std::optional<qcir::QCir>
GraySynthPauliRotationsSynthesisStrategy::synthesize(
    std::vector<PauliRotation> const& rotations) const {
    auto const num_qubits    = rotations.front().n_qubits();
    auto const num_rotations = rotations.size();

    if (num_qubits == 0) {
        return qcir::QCir{0};
    }

    if (num_rotations == 0) {
        return qcir::QCir{num_qubits};
    }

    // checks if all rotations are diagonal
    if (!std::ranges::all_of(rotations, &PauliRotation::is_diagonal)) {
        spdlog::error("GraySynth only supports diagonal rotations");
        return std::nullopt;
    }

    auto frozen_rotations =
        std::unordered_set<std::size_t>{};  // ids to the rotations that
                                            // have been synthesized

    auto copy_rotations = rotations;

    using stack_elem_t =
        std::tuple<
            std::vector<std::size_t>,  // rotation filter
            std::vector<std::size_t>,  // qubit filter
            size_t>;                   // target row
    auto stack = std::vector<stack_elem_t>{};

    stack.emplace_back(
        std::views::iota(0ul, num_rotations) | tl::to<std::vector>(),
        std::views::iota(0ul, num_qubits) | tl::to<std::vector>(),
        SIZE_MAX);

    auto qcir = qcir::QCir{copy_rotations.front().n_qubits()};

    StabilizerTableau final_clifford{num_qubits};

    // generate 0..num_qubits random order
    static auto rng = std::mt19937{42};
    auto random_order =
        std::views::iota(0ul, num_qubits) | tl::to<std::vector>();
    std::ranges::shuffle(random_order, rng);

    while (!stack.empty()) {
        auto const [rotation_filter, qubit_filter, targ] = std::move(stack.back());
        stack.pop_back();
        if (rotation_filter.empty()) continue;
        if (targ != SIZE_MAX) {
            auto ctrls =
                get_control_rows(copy_rotations, rotation_filter, targ);

            apply_cxs(
                std::move(ctrls), targ, mode,
                copy_rotations,
                qcir, final_clifford,
                frozen_rotations, num_rotations, random_order);
        }

        if (qubit_filter.empty()) {
            for (auto col_id : rotation_filter) {
                if (frozen_rotations.contains(col_id)) continue;
                frozen_rotations.insert(col_id);
                DVLAB_ASSERT(
                    targ < num_qubits,
                    "`targ` should be a valid qubit index");
                qcir.append(
                    qcir::PZGate(copy_rotations[col_id].phase()),
                    {targ});
            }
            continue;
        }

        auto const row_id = get_cofactor_row(
            copy_rotations,
            rotation_filter,
            qubit_filter);

        auto const zero_rotations =
            rotation_filter |
            std::views::filter([&](auto const& x) {
                return !copy_rotations[x].pauli_product().is_z_set(row_id);
            }) |
            tl::to<std::vector>();
        auto const one_rotations =
            rotation_filter |
            std::views::filter([&](auto const& x) {
                return copy_rotations[x].pauli_product().is_z_set(row_id);
            }) |
            tl::to<std::vector>();

        stack.emplace_back(
            zero_rotations,
            filter_out_number(qubit_filter, row_id),
            targ);
        stack.emplace_back(
            one_rotations,
            filter_out_number(qubit_filter, row_id),
            targ == SIZE_MAX ? row_id : targ);
    }

    auto const final_clifford_circ = to_qcir(
        final_clifford,
        AGSynthesisStrategy{});

    if (!final_clifford_circ) {
        return std::nullopt;
    }
    qcir.compose(*final_clifford_circ);

    return qcir;
}

namespace {

bool is_valid(PauliRotation const& rotation) {
    if (!rotation.is_diagonal()) {
        return false;
    }

    auto num_z = 0ul;
    for (auto i: std::views::iota(0ul, rotation.n_qubits())) {
        if (rotation.pauli_product().is_z_set(i)) {
            num_z++;
        }
    }
    return num_z == 1;
}

size_t hamming_weight(
    PauliRotation const& rotation) {
    auto const num_qubits = rotation.n_qubits();
    auto num_ones         = 0ul;
    for (auto i : std::views::iota(0ul, num_qubits)) {
        if (rotation.pauli_product().is_z_set(i)) {
            num_ones++;
        }
    }
    return num_ones;
}

size_t qubit_hamming_weight(
    PauliRotation const& rotation) {
    auto const num_qubits = rotation.n_qubits();
    auto num_qubit_has_ones         = 0ul;
    for (auto i : std::views::iota(0ul, num_qubits)) {
        if (rotation.pauli_product().is_z_set(i) || rotation.pauli_product().is_x_set(i)) {
            num_qubit_has_ones++;
        }
    }
    return num_qubit_has_ones;
}

// get the index of the rotation with the minimum number of 1s
// A term of k ones can always be synthesized with k-1 CNOTs
size_t get_best_rotation_idx(
    std::vector<PauliRotation> const& rotations,
    std::string const& strategy = "hamming_weight",
    std::optional<std::unordered_set<size_t>> const& candidates = std::nullopt) {
    auto min_cost = SIZE_MAX;
    auto best_idx = SIZE_MAX;
    
    for (auto const& [idx, rotation] : tl::views::enumerate(rotations)) {
        // Skip if idx is not in candidates (if candidates is provided)
        if (candidates.has_value() && !candidates.value().contains(idx)) {
            continue;
        }
        
        auto const cost = strategy == "hamming_weight" ? 
            hamming_weight(rotation) : 
            qubit_hamming_weight(rotation);
            
        if (cost < min_cost) {
            min_cost = cost;
            best_idx = idx;
        }
    }
    return best_idx;
}

size_t hamming_weight(
    std::vector<PauliRotation> const& rotations,
    size_t q_idx, bool is_Z = true) {
    return std::ranges::count_if(rotations, [&](auto const& rotation) {
        return is_Z ? rotation.pauli_product().is_z_set(q_idx) : rotation.pauli_product().is_x_set(q_idx);
    });
}

size_t hamming_distance(
    std::vector<PauliRotation> const& rotations,
    size_t q1_idx,
    size_t q2_idx) {
    return std::ranges::count_if(rotations, [&](auto const& rotation) {
        return rotation.pauli_product().is_z_set(q1_idx) !=
               rotation.pauli_product().is_z_set(q2_idx);
    });
}

size_t cx_weight(
    std::vector<PauliRotation> const& rotations,
    size_t q1_idx,
    size_t q2_idx) {
    
    size_t x_distance = std::ranges::count_if(rotations, [&](auto const& rotation){
        return rotation.pauli_product().is_x_set(q1_idx) != rotation.pauli_product().is_x_set(q2_idx);
    });

    return x_distance + hamming_distance(rotations, q1_idx, q2_idx);
}

dvlab::Digraph<size_t, int> get_parity_graph(
    std::vector<PauliRotation> const& rotations,
    PauliRotation const& target_rotation,
    std::string const& strategy = "hamming_weight") {
    auto const num_qubits = rotations.front().n_qubits();

    auto g = dvlab::Digraph<size_t, int>{};
    auto qubit_vec = std::vector<size_t>{};

    for (auto i : std::views::iota(0ul, num_qubits)) {
        if (target_rotation.pauli_product().is_z_set(i)) {
            g.add_vertex_with_id(i);
            qubit_vec.push_back(i);
        }
    }
    // get the weight of the edge i if strategy is "hamming_weight"
    // otherwise, get the weight of the cx operation between i and j
    auto const get_weight = [&](size_t i, size_t j) {
        return strategy == "qubit_hamming_weight" 
            ? hamming_weight(rotations, i, true) + hamming_weight(rotations, j, false)
            : hamming_weight(rotations, i, true);
    };

    for (auto const& [i, j] : dvlab::combinations<2>(qubit_vec)) {
        auto const dist = (strategy == "hamming_weight") 
            ? gsl::narrow_cast<int>(hamming_distance(rotations, i, j))
            : gsl::narrow_cast<int>(cx_weight(rotations, i, j));
        auto const weight_i = gsl::narrow_cast<int>(get_weight(i, j));
        auto const weight_j = gsl::narrow_cast<int>(get_weight(j, i));
        g.add_edge(i, j, dist - weight_j - 1);
        g.add_edge(j, i, dist - weight_i - 1);
    }

    return g;
}

dvlab::Digraph<size_t, int> get_dependency_graph(std::vector<PauliRotation> const& rotations, bool check = false) {
    size_t const num_rotations = rotations.size();
    dvlab::Digraph<size_t, int> dag{num_rotations};
    
    // Build dependency edges based on non-commutative pairs
    // size_t counter = 0;
    for (auto i : std::views::iota(0ul, num_rotations)) {
        for (auto j : std::views::iota(i+1, num_rotations)) {
            // if rotations[i] and rotations[j] don't commute, add an edge from i to j
            if (!is_commutative(rotations[i], rotations[j])) {
                // if (i == 0) counter++;
                dag.add_edge(i, j);
                // spdlog::info("Adding edge {} -> {}", i, j);
            }
        }
    }
    // spdlog::info("Number of out degree of 0: {}", counter);
    // Perform transitive reduction to minimize edges
    // dag.transitive_reduction();

    // if (check) {
    //     if (!dag.check_transitive_reduction()) {
    //         spdlog::error("Dependency graph is not transitive reduction");
    //     } else {
    //         spdlog::info("Dependency graph is transitive reduction");
    //     }
    // }
    return dag;
}

// Add the new function before get_dependency_graph
void apply_mst_cxs(dvlab::Digraph<size_t, int> const& mst, size_t root, 
                   std::vector<PauliRotation>& rotations, qcir::QCir& qcir, 
                   StabilizerTableau& final_clifford) {
    
    auto const add_cx = [&](size_t ctrl, size_t targ) {
        for (auto& rot : rotations) {
            rot.cx(ctrl, targ);
        }
        qcir.append(qcir::CXGate(), {ctrl, targ});
        final_clifford.prepend_cx(ctrl, targ);
    };
    // post-order traversal to add CXs
    std::stack<size_t> stack;
    std::vector<size_t> post_order_rev;

    stack.push(root);

    // First phase: collect nodes in post-order
    while (!stack.empty()) {
        auto const v = stack.top();
        stack.pop();
        post_order_rev.push_back(v);

        for (auto const& n : mst.out_neighbors(v)) {
            stack.push(n);
        }
    }
    // Second phase: apply CX gates in reverse post-order
    while (!post_order_rev.empty()) {
        auto const v = post_order_rev.back();
        post_order_rev.pop_back();
        if (mst.in_degree(v) == 1) {
            auto const pred = *mst.in_neighbors(v).begin();
            add_cx(v, pred);
        } else {
            DVLAB_ASSERT(
                mst.in_degree(v) == 0 && v == root,
                "The node with no incoming edges should be the root");
        }
    }
}

}  // namespace

std::optional<qcir::QCir>
MstSynthesisStrategy::synthesize(
    std::vector<PauliRotation> const& rotations) const {
    
    auto const num_qubits    = rotations.front().n_qubits();
    auto const num_rotations = rotations.size();

    if (num_qubits == 0) {
        return qcir::QCir{0};
    }

    if (num_rotations == 0) {
        return qcir::QCir{num_qubits};
    }

    auto copy_rotations = rotations;
    auto qcir = qcir::QCir{copy_rotations.front().n_qubits()};
    StabilizerTableau final_clifford{num_qubits};

    // checks if all rotations are diagonal
    if (!std::ranges::all_of(rotations, &PauliRotation::is_diagonal)) {
        spdlog::error("MST only supports diagonal rotations");
        return std::nullopt;
    }

    while (!copy_rotations.empty()) {
        auto const best_rotation_idx = get_best_rotation_idx(copy_rotations);
        std::swap(copy_rotations[best_rotation_idx], copy_rotations.back());
        auto const best_rotation = std::move(copy_rotations.back());
        copy_rotations.pop_back();

        auto const parity_graph =
            get_parity_graph(copy_rotations, best_rotation);

        auto const [mst, root] =
            dvlab::minimum_spanning_arborescence(parity_graph);
        
        // Use the modularized function
        apply_mst_cxs(mst, root, copy_rotations, qcir, final_clifford);

        // add the rotation at the root
        qcir.append(qcir::PZGate(best_rotation.phase()), {root});
    }

    // synthesize the final clifford
    auto const final_clifford_circ = to_qcir(
        final_clifford,
        AGSynthesisStrategy{});

    if (!final_clifford_circ) {
        return std::nullopt;
    }
    qcir.compose(*final_clifford_circ);

    return qcir;
}

std::optional<qcir::QCir> MSTPauliRotationsSynthesisStrategy::synthesize(std::vector<PauliRotation> const& rotations) const {
    auto const num_qubits    = rotations.front().n_qubits();
    auto const num_rotations = rotations.size();

    if (num_qubits == 0) {
        return qcir::QCir{0};
    }

    if (num_rotations == 0) {
        return qcir::QCir{num_qubits};
    }

    auto copy_rotations = rotations;
    auto qcir = qcir::QCir{copy_rotations.front().n_qubits()};
    StabilizerTableau final_clifford{num_qubits};

    // create the index mapping
    std::vector<size_t> index_mapping(num_rotations);  // col_idx -> vertex_idx
    for (size_t i = 0; i < num_rotations; ++i) {
        index_mapping[i] = i;
    }

    auto dag = get_dependency_graph(copy_rotations);
    // dag.print_graph();

    size_t num_iterations = 0;
    while (!copy_rotations.empty()) {
        // get the first layer rotations
        std::unordered_set<size_t> first_layer_rotations;
        for (auto i : std::views::iota(0ul, copy_rotations.size())) {
            if (dag.in_degree(index_mapping[i]) == 0) {
                first_layer_rotations.insert(i);
            }
        }
        // print first_layer_rotations in a line
        std::string layers = "";
        auto sorted = std::vector<size_t>(first_layer_rotations.begin(), first_layer_rotations.end());
        std::ranges::sort(sorted);
        for (auto const& rot : sorted) {
            layers += std::to_string(index_mapping[rot]) + " ";
        }
        // spdlog::info("First layer rotations");
        // spdlog::info("{}", layers);

        auto const best_rotation_idx = get_best_rotation_idx(copy_rotations, "qubit_hamming_weight", first_layer_rotations);
        

        size_t best_vid = index_mapping[best_rotation_idx];

        auto best_rotation = copy_rotations[best_rotation_idx];
    
        
        // apply Si;Hi if Zi & Xi, apply Hi if -Z & iXi
        for (auto i: std::views::iota(0ul, num_qubits)) {
            if (best_rotation.pauli_product().is_x_set(i)) {
                if (best_rotation.pauli_product().is_z_set(i)) {
                    qcir.append(qcir::SGate(), {i});
                    final_clifford.prepend_sdg(i);
                    for(auto& rot: copy_rotations) { 
                        rot.s(i);
                    }
                }
                qcir.append(qcir::HGate(), {i});
                final_clifford.prepend_h(i);
                for(auto& rot: copy_rotations) {
                    rot.h(i);     
                }
            }
        }
        best_rotation = copy_rotations[best_rotation_idx];
        // best_rotation = std::move(copy_rotations[best_rotation_idx]);
        assert(best_rotation.is_diagonal());
        // copy_rotations.erase(copy_rotations.begin() + best_rotation_idx);
        dag.remove_vertex(best_vid);
        index_mapping.erase(index_mapping.begin() + best_rotation_idx);
        auto const parity_graph = get_parity_graph(copy_rotations, best_rotation, "qubit_hamming_weight");
        auto const [mst, root] = dvlab::minimum_spanning_arborescence(parity_graph);
        
        apply_mst_cxs(mst, root, copy_rotations, qcir, final_clifford);
        assert(is_valid(copy_rotations[best_rotation_idx]));

        copy_rotations.erase(copy_rotations.begin() + best_rotation_idx);
        // add the rotation at the root
        qcir.append(qcir::PZGate(best_rotation.phase()), {root});
    }

    // synthesize the final clifford
    auto const final_clifford_circ = to_qcir(final_clifford, AGSynthesisStrategy{});
    if (!final_clifford_circ) {
        return std::nullopt;
    }
    qcir.compose(*final_clifford_circ);

    return qcir;
}

std::optional<qcir::QCir> HeuristicPauliRotationsSynthesisStrategy::synthesize(std::vector<PauliRotation> const& rotations) const {
    spdlog::error("Heuristic Synthesis Strategy is not implemented yet!!");
    return std::nullopt;
}
/**
 * @brief convert a Pauli rotation to a QCir. This is a naive implementation.
 *
 * @param pauli_rotation
 * @return qcir::QCir
 */
std::optional<qcir::QCir> to_qcir(
    std::vector<PauliRotation> const& pauli_rotations,
    PauliRotationsSynthesisStrategy const& strategy) {
    return strategy.synthesize(pauli_rotations);
}

/**
 * @brief convert a stabilizer tableau and a list of Pauli rotations to a QCir.
 *
 * @param clifford
 * @param pauli_rotations
 * @return qcir::QCir
 */
std::optional<qcir::QCir> to_qcir(Tableau const& tableau, StabilizerTableauSynthesisStrategy const& st_strategy, PauliRotationsSynthesisStrategy const& pr_strategy) {
    qcir::QCir qcir{tableau.n_qubits()};

    for (auto const& subtableau : tableau) {
        if (stop_requested()) {
            return std::nullopt;
        }
        auto const qc_fragment =
            std::visit(
                dvlab::overloaded{
                    [&st_strategy](StabilizerTableau const& st) { return to_qcir(st, st_strategy); },
                    [&pr_strategy](std::vector<PauliRotation> const& pr) { return to_qcir(pr, pr_strategy); }},
                subtableau);
        if (!qc_fragment) {
            return std::nullopt;
        }
        qcir.compose(*qc_fragment);
    }

    return qcir;
}

}  // namespace qsyn::experimental
