/****************************************************************************
  PackageName  [ duostra ]
  Synopsis     [ Define class Duostra member functions ]
  Author       [ Chin-Yi Cheng, Chien-Yi Yang, Ren-Chu Wang, Yi-Hsiang Kuo ]
  Paper        [ https://arxiv.org/abs/2210.01306 ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./duostra.hpp"

#include <spdlog/spdlog.h>

#include "./placer.hpp"
#include "duostra/mapping_eqv_checker.hpp"
#include "qcir/basic_gate_type.hpp"
#include "qcir/qcir.hpp"
#include "qsyn/qsyn_type.hpp"

extern bool stop_requested();

using namespace qsyn::qcir;

namespace qsyn::duostra {

/**
 * @brief Construct a new Duostra Mapper object
 *
 * @param cir
 * @param dev
 * @param check
 * @param tqdm
 * @param silent
 */
Duostra::Duostra(
    QCir* cir,
    Device dev,
    DuostraConfig const& config,
    DuostraExecutionOptions const& exe_opts)
    : _device(std::move(dev)),
      _config{config},
      _check(exe_opts.verify_result),
      _tqdm{!exe_opts.silent && exe_opts.use_tqdm},
      _silent{exe_opts.silent},
      _logical_circuit{std::make_shared<qcir::QCir>(*cir)} {}

/**
 * @brief Main flow of Duostra mapper
 *
 * @return size_t
 */
bool Duostra::map(bool use_device_as_placement) {
    std::unique_ptr<CircuitTopology> topo;
    topo            = make_unique<CircuitTopology>(_logical_circuit);
    auto check_topo = topo->clone();
    auto check_device(_device);

    spdlog::info("Creating device...");
    if (topo->get_num_qubits() > _device.get_num_qubits()) {
        spdlog::error("Number of logical qubits are larger than the device!!");
        return false;
    }

    std::vector<QubitIdType> assign;
    if (!use_device_as_placement) {
        spdlog::info("Calculating Initial Placement...");
        auto placer = get_placer(_config.placer_type);
        assign      = placer->place_and_assign(_device);
    }
    // scheduler
    spdlog::info("Creating Scheduler...");
    auto scheduler = get_scheduler(_config, std::move(topo), _tqdm);

    // router
    spdlog::info("Creating Router...");
    auto cost_strategy =
        (_config.scheduler_type == SchedulerType::greedy)
            ? Router::CostStrategyType::end
            : Router::CostStrategyType::start;
    auto router = std::make_unique<Router>(
        std::move(_device),
        _config.router_type,
        cost_strategy,
        _config.tie_breaking_strategy);

    // routing
    if (!_silent) {
        fmt::println("Routing...");
    }
    _device = scheduler->assign_gates_and_sort(std::move(router));
    if (stop_requested()) {
        spdlog::warn("Warning: mapping interrupted");
        return false;
    }

    assert(scheduler->is_sorted());
    // assert(scheduler->get_order().size() == _logical_circuit->get_gates().size());

    for (auto const& [gate, _] : scheduler->get_operations())
        _result.emplace_back(gate);

    // store_order_info(scheduler->get_order());
    build_circuit_by_result();

    if (_check) {
        if (!_silent) {
            fmt::println("Checking...");
            fmt::println("");
        }
        auto checker = MappingEquivalenceChecker(
            _physical_circuit.get(),
            _logical_circuit.get(),
            check_device,
            _config.placer_type);
        if (!checker.check()) {
            return false;
        }
    }

    if (!_silent) {
        fmt::println("Duostra Result: ");
        fmt::println("");
        fmt::println("Scheduler:      {}", get_scheduler_type_str(_config.scheduler_type));
        fmt::println("Router:         {}", get_router_type_str(_config.router_type));
        fmt::println("Placer:         {}", get_placer_type_str(_config.placer_type));
        fmt::println("");
        fmt::println("Mapping Depth:  {}", scheduler->get_final_cost());
        fmt::println("Total Time:     {}", scheduler->get_total_time());
        fmt::println("#SWAP:          {}", scheduler->get_num_swaps());
        fmt::println("");
    }

    return true;
}

/**
 * @brief Convert index to full information of gate
 *
 * @param order
 */
// void Duostra::store_order_info(std::vector<size_t> const& order) {
//     // for (auto const& gate_id : order) {
//     //     auto const& g = _logical_circuit->get_gate(gate_id);
//     //     _order.emplace_back(*g);
//     // }
// }

/**
 * @brief Construct physical QCir by operation
 *
 */
void Duostra::build_circuit_by_result() {
    _physical_circuit->add_qubits(_device.get_num_qubits());
    for (auto const& operation : _result) {
        auto qubits = operation.get_qubits();
        QubitIdList qu;
        qu.emplace_back(qubits[0]);
        if (qubits[1] != max_qubit_id) {
            qu.emplace_back(qubits[1]);
        }
        if (operation.get_operation().is<SwapGate>()) {
            // NOTE - Decompose SWAP into three CX
            QubitIdList qu_reverse;
            qu_reverse.emplace_back(qubits[1]);
            qu_reverse.emplace_back(qubits[0]);
            _physical_circuit->append(CXGate(), qu);
            _physical_circuit->append(CXGate(), qu_reverse);
            _physical_circuit->append(CXGate(), qu);
        } else {
            _physical_circuit->append(operation.get_operation(), qu);
        }
    }
}

}  // namespace qsyn::duostra
