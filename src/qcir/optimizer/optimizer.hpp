/****************************************************************************
  PackageName  [ qcir/optimizer ]
  Synopsis     [ Define class Optimizer structure ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#pragma once

#include <cstddef>
#include <vector>

#include "qcir/basic_gate_type.hpp"
#include "qcir/qcir_gate.hpp"
#include "qsyn/qsyn_type.hpp"

namespace dvlab {

class Phase;

}

namespace qsyn::qcir {

class QCir;

class Optimizer {
public:
    Optimizer() {}

    void reset(QCir const& qcir);

    // Predicate function && Utils
    bool is_single_z_rotation(QCirGate const& g) const;
    bool is_single_x_rotation(QCirGate const& g) const;
    std::optional<size_t> get_available_z_rotation(QubitIdType t) const;

    // basic optimization
    struct BasicOptimizationConfig {
        bool doSwap          = true;
        size_t maxIter       = 1000;
        bool printStatistics = false;
    };
    std::optional<QCir> basic_optimization(QCir const& qcir, BasicOptimizationConfig const& config);
    QCir parse_forward(QCir const& qcir, bool do_minimize_czs, BasicOptimizationConfig const& config);
    QCir parse_backward(QCir const& qcir, bool do_minimize_czs, BasicOptimizationConfig const& config);
    bool parse_gate(QCirGate& gate, bool do_swap, bool do_minimize_czs);

    // trivial optimization
    std::optional<QCir> trivial_optimization(QCir const& qcir);

private:
    size_t _iter = 0;
    std::vector<QCirGate> _storage;
    std::vector<std::vector<size_t>> _gates;
    std::vector<std::vector<size_t>> _available_gates;
    std::vector<bool> _qubit_available;

    std::vector<QubitIdType> _permutation;

    std::vector<bool> _hs;
    std::vector<bool> _xs;
    std::vector<bool> _zs;
    std::vector<std::pair<QubitIdType, QubitIdType>> _swaps;

    struct Statistics {
        size_t FUSE_PHASE    = 0;
        size_t X_CANCEL      = 0;
        size_t CNOT_CANCEL   = 0;
        size_t CZ_CANCEL     = 0;
        size_t HS_EXCHANGE   = 0;
        size_t CRZ_TRANSFORM = 0;
        size_t DO_SWAP       = 0;
        size_t CZ2CX         = 0;
        size_t CX2CZ         = 0;
    } _statistics;

    // Utils
    enum class ElementType : std::uint8_t {
        h,
        x,
        z
    };
    void _toggle_element(ElementType type, QubitIdType element);
    void _swap_element(ElementType type, QubitIdType e1, QubitIdType e2);
    static std::vector<size_t> _compute_stats(QCir const& circuit);

    QCir _parse_once(QCir const& qcir, bool reversed, bool do_minimize_czs, BasicOptimizationConfig const& config);

    // basic optimization subroutines

    void _permute_gates(QCirGate& gate);

    void _match_hadamards(QCirGate const& gate);
    void _match_xs(QCirGate const& gate);
    void _match_z_rotations(QCirGate& gate);
    void _match_czs(QCirGate& gate, bool do_swap, bool do_minimize_czs);
    void _match_cxs(QCirGate const& gate, bool do_swap, bool do_minimize_czs);

    void _add_hadamard(QubitIdType target, bool erase);
    bool _replace_cx_and_cz_with_s_and_cx(QubitIdType t1, QubitIdType t2);
    void _add_cz(QubitIdType t1, QubitIdType t2, bool do_minimize_czs);
    void _add_cx(QubitIdType t1, QubitIdType t2, bool do_swap);
    void _add_single_z_rotation_gate(QubitIdType target, dvlab::Phase ph);

    QCir _build_from_storage(size_t n_qubits, bool reversed);

    std::vector<std::pair<QubitIdType, QubitIdType>> _get_swap_path();

    // trivial optimization subroutines

    void _fuse_z_phase(QCir& qcir, QCirGate* prev_gate, QCirGate* gate);
    void _fuse_x_phase(QCir& qcir, QCirGate* prev_gate, QCirGate* gate);
    void _partial_zx_optimization(QCir& qcir);

    size_t _store_x(QubitIdType qubit) {
        _storage.emplace_back(_storage.size(), XGate(), QubitIdList{qubit});
        return _storage.size() - 1;
    }

    size_t _store_h(QubitIdType qubit) {
        _storage.emplace_back(_storage.size(), HGate(), QubitIdList{qubit});
        return _storage.size() - 1;
    }

    size_t _store_s(QubitIdType qubit) {
        _storage.emplace_back(_storage.size(), SGate(), QubitIdList{qubit});
        return _storage.size() - 1;
    }

    size_t _store_sdg(QubitIdType qubit) {
        _storage.emplace_back(_storage.size(), SdgGate(), QubitIdList{qubit});
        return _storage.size() - 1;
    }

    size_t _store_cx(QubitIdType ctrl, QubitIdType targ) {
        _storage.emplace_back(_storage.size(), CXGate(), QubitIdList{ctrl, targ});
        return _storage.size() - 1;
    }

    size_t _store_cz(QubitIdType ctrl, QubitIdType targ) {
        _storage.emplace_back(_storage.size(), CZGate(), QubitIdList{ctrl, targ});
        return _storage.size() - 1;
    }

    size_t _store_single_z_rotation_gate(QubitIdType target, dvlab::Phase ph) {
        _storage.emplace_back(_storage.size(), PZGate(ph), QubitIdList{target});
        return _storage.size() - 1;
    }
};

void phase_teleport(QCir& qcir);
void optimize_2q_count(QCir& qcir,
                       double hadamard_insertion_ratio,
                       size_t max_lc_unfusions,
                       size_t max_pv_unfusions);

}  // namespace qsyn::qcir
