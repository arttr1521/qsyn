/****************************************************************************
  PackageName  [ duostra ]
  Synopsis     [ Define class Router structure ]
  Author       [ Chin-Yi Cheng, Chien-Yi Yang, Ren-Chu Wang, Yi-Hsiang Kuo ]
  Paper        [ https://arxiv.org/abs/2210.01306 ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#pragma once

#include <queue>

#include "./duostra_def.hpp"
#include "device/device.hpp"
#include "qcir/qcir_gate.hpp"
#include "qsyn/qsyn_type.hpp"

namespace qsyn::duostra {

class AStarNode {
public:
    friend class AStarComp;
    AStarNode(size_t cost, QubitIdType id, bool source);

    auto get_source() const { return _source; }
    auto get_id() const { return _id; }
    auto get_cost() const { return _estimated_cost; }

private:
    size_t _estimated_cost;
    QubitIdType _id;
    bool _source;  // false q0 propagate, true q1 propagate
};

class AStarComp {
public:
    bool operator()(AStarNode const& a, AStarNode const& b) {
        if (a._estimated_cost == b._estimated_cost)
            return a._id > b._id;
        return a._estimated_cost > b._estimated_cost;
    }
};

class Router {
public:
    using Device        = qsyn::device::Device;
    using PhysicalQubit = qsyn::device::PhysicalQubit;

    enum CostStrategyType : std::uint8_t {
        start,
        end
    };

    using PriorityQueue = std::priority_queue<AStarNode, std::vector<AStarNode>, AStarComp>;
    Router(Device device, RouterType type, CostStrategyType cost_strategy, MinMaxOptionType tie_breaking_strategy);

    std::unique_ptr<Router> clone() const;

    auto& get_device() { return _device; }
    auto const& get_device() const { return _device; }

    size_t get_gate_cost(qcir::QCirGate const& gate, MinMaxOptionType min_max, size_t apsp_coeff);
    bool is_executable(qcir::QCirGate const& gate);

    // Main Router function
    GateInfo execute_single(qcir::QCirGate const& gate, QubitIdType q);
    std::vector<GateInfo> duostra_routing(qcir::QCirGate const& gate, std::tuple<QubitIdType, QubitIdType> qubit_pair, MinMaxOptionType tie_breaking_strategy);
    std::vector<GateInfo> apsp_routing(qcir::QCirGate const& gate, std::tuple<QubitIdType, QubitIdType> qs, MinMaxOptionType tie_breaking_strategy);
    std::vector<GateInfo> assign_gate(qcir::QCirGate const& gate);

private:
    MinMaxOptionType _tie_breaking_strategy;
    Device _device;
    std::vector<QubitIdType> _logical_to_physical;
    bool _apsp : 1;
    bool _duostra : 1;

    void _initialize();
    std::tuple<QubitIdType, QubitIdType> _get_physical_qubits(qcir::QCirGate const& gate) const;

    std::tuple<bool, QubitIdType> _touch_adjacency(PhysicalQubit& qubit, PriorityQueue& pq, bool source);  // return <if touch target, target id>, swtch: false q0 propagate, true q1 propagate
    std::vector<GateInfo> _traceback(qcir::QCirGate const& gate, PhysicalQubit& q0, PhysicalQubit& q1, PhysicalQubit& t0, PhysicalQubit& t1, bool swap_ids);
};

}  // namespace qsyn::duostra
