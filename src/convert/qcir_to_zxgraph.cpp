/****************************************************************************
  PackageName  [ qsyn ]
  Synopsis     [ Define conversion from QCir to ZXGraph ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./qcir_to_zxgraph.hpp"

#include <spdlog/spdlog.h>

#include <cstddef>
#include <gsl/narrow>
#include <ranges>
#include <tl/enumerate.hpp>

#include "qcir/basic_gate_type.hpp"
#include "qcir/qcir.hpp"
#include "qcir/qcir_gate.hpp"
#include "util/phase.hpp"
#include "util/rational.hpp"
#include "util/util.hpp"
#include "zx/zx_def.hpp"
#include "zx/zxgraph.hpp"
#include "zx/zxgraph_action.hpp"

extern bool stop_requested();

namespace qsyn {

using zx::ZXVertex, zx::ZXGraph, zx::VertexType, zx::EdgeType;

using qcir::QCir;

namespace {

dvlab::Phase get_gadget_phase(dvlab::Phase const& phase, size_t n_qubits) {
    return phase * dvlab::Rational(1, static_cast<int>(std::pow(2, gsl::narrow<double>(n_qubits) - 1)));
}

enum class RotationAxis : std::uint8_t {
    x,
    y,
    z
};

std::pair<std::vector<ZXVertex*>, ZXVertex*>
create_multi_control_backbone(ZXGraph& g, size_t num_qubits, RotationAxis ax) {
    std::vector<ZXVertex*> controls;
    ZXVertex* target  = nullptr;
    auto target_qubit = num_qubits - 1;
    for (auto qubit : std::views::iota(0ul, num_qubits)) {
        ZXVertex* in  = g.add_input(qubit);
        ZXVertex* v   = g.add_vertex(VertexType::z, dvlab::Phase{}, static_cast<float>(qubit));
        ZXVertex* out = g.add_output(qubit);
        if (ax == RotationAxis::z || qubit != target_qubit) {
            g.add_edge(in, v, EdgeType::simple);
            g.add_edge(v, out, EdgeType::simple);
        } else {
            g.add_edge(in, v, EdgeType::hadamard);
            g.add_edge(v, out, EdgeType::hadamard);
            if (ax == RotationAxis::y) {
                auto const buffer1 = zx::add_identity_vertex(
                    g, v->get_id(), in->get_id(),
                    VertexType::z, EdgeType::hadamard);
                auto const buffer2 = zx::add_identity_vertex(
                    g, v->get_id(), out->get_id(),
                    VertexType::z, EdgeType::hadamard);

                assert(buffer1.has_value() && buffer2.has_value());

                g[*buffer1]->phase() = dvlab::Phase(-1, 2);
                g[*buffer2]->phase() = dvlab::Phase(1, 2);
            }
        }
        if (qubit != target_qubit)
            controls.emplace_back(v);
        else
            target = v;
    }

    assert(target != nullptr);

    return {controls, target};
}

void create_multi_control_r_gate_gadgets(ZXGraph& g, std::vector<ZXVertex*> const& controls, ZXVertex* target, dvlab::Phase const& phase) {
    target->phase() = phase;
    for (size_t k = 1; k <= controls.size(); k++) {
        for (auto& combination : dvlab::combinations(controls,
                                                     k,
                                                     std::ranges::less{},
                                                     [](ZXVertex* v) { return v->get_id(); })) {
            combination.emplace_back(target);
            g.add_gadget((combination.size() % 2) ? phase : -phase, combination);
        }
    }
}

void create_multi_control_p_gate_gadgets(ZXGraph& g, std::vector<ZXVertex*> const& vertices, dvlab::Phase const& phase) {
    for (auto& v : vertices) {
        v->phase() = phase;
    }
    for (size_t k = 2; k <= vertices.size(); k++) {
        for (auto& combination : dvlab::combinations(vertices,
                                                     k,
                                                     std::ranges::less{},
                                                     [](ZXVertex* v) { return v->get_id(); })) {
            g.add_gadget((combination.size() % 2) ? phase : -phase, combination);
        }
    }
}

ZXGraph create_mcr_zx_form(size_t num_qubits, dvlab::Phase const& ph, RotationAxis ax) {
    ZXGraph g;
    auto const gadget_phase = get_gadget_phase(ph, num_qubits);

    auto const [controls, target] = create_multi_control_backbone(g, num_qubits, ax);

    create_multi_control_r_gate_gadgets(g, controls, target, gadget_phase);

    return g;
}

ZXGraph create_mcp_zx_form(size_t num_qubits, dvlab::Phase const& ph, RotationAxis ax) {
    ZXGraph g;
    auto const gadget_phase = get_gadget_phase(ph, num_qubits);

    auto [vertices, target] = create_multi_control_backbone(g, num_qubits, ax);
    vertices.emplace_back(target);

    create_multi_control_p_gate_gadgets(g, vertices, gadget_phase);

    return g;
}

/**
 * @brief Map single qubit gate to ZXGraph
 *
 * @param vt
 * @param ph
 * @return ZXGraph
 */
ZXGraph create_single_vertex_zx_form(VertexType vt, dvlab::Phase const& ph) {
    ZXGraph g;

    ZXVertex* in  = g.add_input(0);
    ZXVertex* v   = g.add_vertex(vt, ph, static_cast<float>(0));
    ZXVertex* out = g.add_output(0);
    g.add_edge(in, v, EdgeType::simple);
    g.add_edge(v, out, EdgeType::simple);

    return g;
}

// Double or More Qubit Gate

/**
 * @brief get ZXGraph of CX
 *
 * @return ZXGraph
 */
ZXGraph create_cx_zx_form() {
    ZXGraph g;

    ZXVertex* in_ctrl  = g.add_input(0);
    ZXVertex* in_targ  = g.add_input(1);
    ZXVertex* ctrl     = g.add_vertex(VertexType::z, dvlab::Phase(0), 0);
    ZXVertex* targ_x   = g.add_vertex(VertexType::x, dvlab::Phase(0), 1);
    ZXVertex* out_ctrl = g.add_output(0);
    ZXVertex* out_targ = g.add_output(1);
    g.add_edge(in_ctrl, ctrl, EdgeType::simple);
    g.add_edge(ctrl, out_ctrl, EdgeType::simple);
    g.add_edge(in_targ, targ_x, EdgeType::simple);
    g.add_edge(targ_x, out_targ, EdgeType::simple);
    g.add_edge(ctrl, targ_x, EdgeType::simple);

    return g;
}

// /**
//  * @brief Cet ZXGraph of CCX.
//  *        Decomposed into 21 vertices (6X + 6Z + 4T + 3Tdg + 2H)
//  *
//  * @return ZXGraph
//  */
// ZXGraph create_ccx_zx_form(QCirGate* gate) {
//     ZXGraph g;
//     auto ctrl_qubit_2 = gate->get_qubit(0);
//     auto ctrl_qubit_1 = gate->get_qubit(1);
//     auto targ_qubit   = gate->get_qubit(2);
//     std::vector<std::pair<std::pair<VertexType, dvlab::Phase>, QubitIdType>> vertices_info;
//     std::vector<std::pair<std::pair<size_t, size_t>, EdgeType>> adj_pair;
//     std::vector<float> vertices_col;
//     std::vector<ZXVertex*> vertices_list = {};
//     ZXVertex* in_ctrl_1                  = nullptr;
//     ZXVertex* in_ctrl_2                  = nullptr;
//     ZXVertex* in_targ                    = nullptr;
//     ZXVertex* out_ctrl_1                 = nullptr;
//     ZXVertex* out_ctrl_2                 = nullptr;
//     ZXVertex* out_targ                   = nullptr;

//     vertices_info = {{{VertexType::z, dvlab::Phase(1, 4)}, targ_qubit}, {{VertexType::z, dvlab::Phase(1, 4)}, ctrl_qubit_2}, {{VertexType::z, dvlab::Phase(1, 4)}, ctrl_qubit_1}, {{VertexType::z, dvlab::Phase(1, 4)}, -2}, {{VertexType::z, dvlab::Phase(0)}, -1}, {{VertexType::z, dvlab::Phase(-1, 4)}, -2}, {{VertexType::z, dvlab::Phase(0)}, -1}, {{VertexType::z, dvlab::Phase(-1, 4)}, -2}, {{VertexType::z, dvlab::Phase(0)}, -1}, {{VertexType::z, dvlab::Phase(-1, 4)}, -2}, {{VertexType::z, dvlab::Phase(0)}, -1}};
//     adj_pair      = {{{0, 4}, EdgeType::hadamard}, {{0, 6}, EdgeType::hadamard}, {{0, 8}, EdgeType::hadamard}, {{1, 4}, EdgeType::hadamard}, {{1, 6}, EdgeType::hadamard}, {{1, 10}, EdgeType::hadamard}, {{2, 4}, EdgeType::hadamard}, {{2, 8}, EdgeType::hadamard}, {{2, 10}, EdgeType::hadamard}, {{3, 4}, EdgeType::hadamard}, {{5, 6}, EdgeType::hadamard}, {{7, 8}, EdgeType::hadamard}, {{9, 10}, EdgeType::hadamard}};
//     vertices_col  = {5, 5, 5, 1, 1, 2, 2, 3, 3, 4, 4};

//     in_ctrl_1 = g.add_input(ctrl_qubit_1, 0);
//     in_ctrl_2 = g.add_input(ctrl_qubit_2, 0);
//     in_targ   = g.add_input(targ_qubit, 0);
//     for (size_t i = 0; i < vertices_info.size(); i++) {
//         vertices_list.emplace_back(g.add_vertex(vertices_info[i].first.first, vertices_info[i].first.second, static_cast<float>(vertices_info[i].second), vertices_col[i]));
//     };
//     out_ctrl_1 = g.add_output(ctrl_qubit_1, 6);
//     out_ctrl_2 = g.add_output(ctrl_qubit_2, 6);
//     out_targ   = g.add_output(targ_qubit, 6);
//     g.add_edge(in_ctrl_1, vertices_list[2], EdgeType::simple);
//     g.add_edge(in_ctrl_2, vertices_list[1], EdgeType::simple);
//     g.add_edge(in_targ, vertices_list[0], EdgeType::hadamard);
//     g.add_edge(out_ctrl_1, vertices_list[2], EdgeType::simple);
//     g.add_edge(out_ctrl_2, vertices_list[1], EdgeType::simple);
//     g.add_edge(out_targ, vertices_list[0], EdgeType::hadamard);

//     return g;
// }

ZXGraph create_swap_zx_form() {
    ZXGraph g;
    auto i0 = g.add_input(0, 0);
    auto o0 = g.add_output(0, 1);
    auto i1 = g.add_input(1, 0);
    auto o1 = g.add_output(1, 1);
    g.add_edge(i0, o1, EdgeType::simple);
    g.add_edge(i1, o0, EdgeType::simple);

    return g;
}

ZXGraph create_ecr_zx_form() {
    ZXGraph g;
    auto i0 = g.add_input(0, 0);
    auto o0 = g.add_output(0, 3);
    auto i1 = g.add_input(1, 0);
    auto o1 = g.add_output(1, 3);
    auto s0 = g.add_vertex(VertexType::z, dvlab::Phase(1, 2), 0, 1);
    auto v1 = g.add_vertex(VertexType::x, dvlab::Phase(1, 2), 1, 1);
    auto x0 = g.add_vertex(VertexType::x, dvlab::Phase(1), 0, 2);

    g.add_edge(i0, s0, EdgeType::simple);
    g.add_edge(s0, x0, EdgeType::simple);
    g.add_edge(x0, o0, EdgeType::simple);
    g.add_edge(i1, v1, EdgeType::simple);
    g.add_edge(v1, o1, EdgeType::simple);
    g.add_edge(s0, v1, EdgeType::simple);

    return g;
}

/**
 * @brief Get ZXGraph of CZ
 *
 * @return ZXGraph
 */
ZXGraph create_cz_zx_form() {
    ZXGraph g;

    ZXVertex* in_ctrl  = g.add_input(0);
    ZXVertex* in_targ  = g.add_input(1);
    ZXVertex* ctrl     = g.add_vertex(VertexType::z, dvlab::Phase(0), 0);
    ZXVertex* targ_z   = g.add_vertex(VertexType::z, dvlab::Phase(0), 1);
    ZXVertex* out_ctrl = g.add_output(0);
    ZXVertex* out_targ = g.add_output(1);
    g.add_edge(in_ctrl, ctrl, EdgeType::simple);
    g.add_edge(ctrl, out_ctrl, EdgeType::simple);
    g.add_edge(in_targ, targ_z, EdgeType::simple);
    g.add_edge(targ_z, out_targ, EdgeType::simple);
    g.add_edge(ctrl, targ_z, EdgeType::hadamard);

    return g;
}

/**
 * @brief Get ZXGraph of SY = S。SX。Sdg
 *
 * @return ZXGraph
 */
ZXGraph create_ry_zx_form(dvlab::Phase const& ph) {
    ZXGraph g;

    ZXVertex* in  = g.add_input(0);
    ZXVertex* sdg = g.add_vertex(VertexType::z, dvlab::Phase(-1, 2), static_cast<float>(0));
    ZXVertex* rx  = g.add_vertex(VertexType::x, ph, static_cast<float>(0));
    ZXVertex* s   = g.add_vertex(VertexType::z, dvlab::Phase(1, 2), static_cast<float>(0));
    ZXVertex* out = g.add_output(0);
    g.add_edge(in, sdg, EdgeType::simple);
    g.add_edge(sdg, rx, EdgeType::simple);
    g.add_edge(rx, s, EdgeType::simple);
    g.add_edge(s, out, EdgeType::simple);

    return g;
}

}  // namespace

template <>
std::optional<ZXGraph> to_zxgraph(qcir::IdGate const& /* op */) {
    ZXGraph g;
    ZXVertex* in  = g.add_input(0);
    ZXVertex* out = g.add_output(0);
    g.add_edge(in, out, EdgeType::simple);
    return g;
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::HGate const& /* op */) {
    return create_single_vertex_zx_form(VertexType::h_box, dvlab::Phase(1));
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::SwapGate const& /* op */) {
    return create_swap_zx_form();
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::ECRGate const& /* op */) {
    return create_ecr_zx_form();
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::PZGate const& op) {
    return create_single_vertex_zx_form(VertexType::z, op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::PXGate const& op) {
    return create_single_vertex_zx_form(VertexType::x, op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::PYGate const& op) {
    return create_ry_zx_form(op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::RZGate const& op) {
    return create_single_vertex_zx_form(VertexType::z, op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::RXGate const& op) {
    return create_single_vertex_zx_form(VertexType::x, op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::RYGate const& op) {
    return create_ry_zx_form(op.get_phase());
}

template <>
std::optional<ZXGraph> to_zxgraph(qcir::ControlGate const& op) {
    auto const& target_op = op.get_target_operation();

    if (auto const px = target_op.get_underlying_if<qcir::PXGate>()) {
        if (op.get_num_qubits() == 2 && px->get_phase() == Phase(1)) {
            return create_cx_zx_form();
        }
        return create_mcp_zx_form(op.get_num_qubits(), px->get_phase(), RotationAxis::x);
    }

    if (auto const py = target_op.get_underlying_if<qcir::PYGate>()) {
        return create_mcp_zx_form(op.get_num_qubits(), py->get_phase(), RotationAxis::y);
    }

    if (auto const pz = target_op.get_underlying_if<qcir::PZGate>()) {
        if (op.get_num_qubits() == 2 && pz->get_phase() == Phase(1)) {
            return create_cz_zx_form();
        }

        return create_mcp_zx_form(op.get_num_qubits(), pz->get_phase(), RotationAxis::z);
    }

    if (auto const rx = target_op.get_underlying_if<qcir::RXGate>()) {
        return create_mcr_zx_form(op.get_num_qubits(), rx->get_phase(), RotationAxis::x);
    }

    if (auto const ry = target_op.get_underlying_if<qcir::RYGate>()) {
        return create_mcr_zx_form(op.get_num_qubits(), ry->get_phase(), RotationAxis::y);
    }

    if (auto const rz = target_op.get_underlying_if<qcir::RZGate>()) {
        return create_mcr_zx_form(op.get_num_qubits(), rz->get_phase(), RotationAxis::z);
    }

    return std::nullopt;
}

std::optional<ZXGraph> to_zxgraph(qcir::QCirGate const& gate) {
    return to_zxgraph(gate.get_operation());
}

/**
 * @brief Mapping QCir to ZXGraph
 */
template <>
std::optional<ZXGraph> to_zxgraph(QCir const& qcir) {
    if (qcir.is_empty()) {
        spdlog::error("QCir is empty!!");
        return std::nullopt;
    }
    auto const times = qcir.calculate_gate_times();

    ZXGraph graph;
    spdlog::debug("Add boundaries");
    for (auto i : std::views::iota(0ul, qcir.get_num_qubits())) {
        ZXVertex* input  = graph.add_input(i);
        ZXVertex* output = graph.add_output(i);
        graph.add_edge(input, output, EdgeType::simple);
    }

    for (auto const& gate : qcir.get_gates()) {
        if (stop_requested()) {
            spdlog::warn("Conversion interrupted.");
            return std::nullopt;
        }
        spdlog::debug("Gate {} ({})", gate->get_id(), gate->get_operation().get_repr());

        auto tmp = to_zxgraph(*gate);

        if (!tmp) {
            spdlog::error("Conversion of Gate {} ({}) to ZXGraph is not supported yet!!", gate->get_id(), gate->get_operation().get_repr());
            return std::nullopt;
        }

        for (auto& v : tmp->get_vertices()) {
            v->set_col(v->get_col() + static_cast<float>(times.at(gate->get_id())));
        }

        graph.concatenate(*std::move(tmp), gate->get_qubits());
    }

    auto const max_col = std::ranges::max(graph.get_outputs() | std::views::transform([&graph](ZXVertex* v) { return graph.get_first_neighbor(v).first->get_col(); }));
    for (auto& v : graph.get_outputs()) {
        v->set_col(max_col + 1);
    }

    if (stop_requested()) {
        spdlog::warn("Conversion interrupted.");
        return std::nullopt;
    }

    return graph;
}

}  // namespace qsyn
