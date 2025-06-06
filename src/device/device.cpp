/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Define class Device, Topology, and Operation functions ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "device/device.hpp"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <gsl/narrow>
#include <ranges>
#include <string>
#include <tl/enumerate.hpp>
#include <tl/to.hpp>
#include <utility>

#include "qcir/basic_gate_type.hpp"
#include "qcir/qcir_gate.hpp"
#include "qsyn/qsyn_type.hpp"
#include "util/dvlab_string.hpp"
#include "util/util.hpp"

using namespace qsyn::qcir;

template <>
struct fmt::formatter<qsyn::device::PhysicalQubit> {
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    template <typename FormatContext>
    auto format(qsyn::device::PhysicalQubit const& q, FormatContext& ctx) {
        return fmt::format_to(ctx.out(), "Q{:>2}, logical: {:>2}, lock until {}", q.get_id(), q.get_logical_qubit(), q.get_occupied_time());
    }
};

namespace qsyn::device {

/**
 * @brief TEMPORARY SOLUTION! get gate delay as used by the Duostra algorithm
 *        This function should be rewritten to use the actual device information
 *
 * @param inst
 * @return size_t
 */
size_t Device::get_delay(qcir::QCirGate const& inst) const {
    if (inst.get_operation().get_type() == "swap") {
        return 6;
    } else if (inst.get_qubits().size() == 1) {
        return 1;
    } else if (inst.get_qubits().size() == 2) {
        return 2;
    } else {
        return 5;
    }
}

// SECTION - Class Topology Member Functions

/**
 * @brief Get the information of a single adjacency pair
 *
 * @param a Id of first qubit
 * @param b Id of second qubit
 * @return Info&
 */
DeviceInfo const& Topology::get_adjacency_pair_info(size_t a, size_t b) {
    if (a > b) std::swap(a, b);
    return _adjacency_info[std::make_pair(a, b)];
}

/**
 * @brief Get the information of a qubit
 *
 * @param a
 * @return const Info&
 */
DeviceInfo const& Topology::get_qubit_info(size_t a) {
    return _qubit_info[a];
}

/**
 * @brief Add adjacency information of (a,b)
 *
 * @param a Id of first qubit
 * @param b Id of second qubit
 * @param info Information of this pair
 */
void Topology::add_adjacency_info(size_t a, size_t b, DeviceInfo info) {
    if (a > b) std::swap(a, b);
    _adjacency_info[std::make_pair(a, b)] = info;
}

/**
 * @brief Add qubit information
 *
 * @param a
 * @param info
 */
void Topology::add_qubit_info(size_t a, DeviceInfo info) {
    _qubit_info[a] = info;
}

/**
 * @brief Resize distance and predecessor lists
 *
 */
void Topology::resize_to_num_qubit() {
    _distance.resize(_num_qubit);
    _predecessor.resize(_num_qubit);

    for (size_t i = 0; i < _num_qubit; i++) {
        _distance[i].resize(_num_qubit);
        _predecessor[i].resize(_num_qubit, max_qubit_id);
    }
}

/**
 * @brief Initialize predecessor and distance lists
 *
 * @param adjacency_matrix
 * @param qubit_list
 */
void Topology::_init_predecessor_and_distance(const std::vector<std::vector<QubitIdType>>& adjacency_matrix, const std::vector<PhysicalQubit>& qubit_list) {
    resize_to_num_qubit();
    for (size_t i = 0; i < _num_qubit; i++) {
        for (size_t j = 0; j < _num_qubit; j++) {
            _distance[i][j] = adjacency_matrix[i][j];
            if (_distance[i][j] != 0 && _distance[i][j] != _max_dist) {
                _predecessor[i][j] = qubit_list[i].get_id();
            }
        }
    }
}

/**
 * @brief Set weight of edge used in Floyd-Warshall
 *
 * @param adjacency_matrix
 * @param qubit_list
 */
void Topology::_set_weight(std::vector<std::vector<QubitIdType>>& adjacency_matrix, const std::vector<PhysicalQubit>& qubit_list) const {
    assert(adjacency_matrix.size() == _num_qubit);
    for (size_t i = 0; i < _num_qubit; i++) {
        for (auto const& adj : qubit_list[i].get_adjacencies()) {
            adjacency_matrix[i][adj] = 1;
        }
    }
}

/**
 * @brief Floyd-Warshall Algorithm. Solve All Pairs Shortest Path (APSP)
 *
 * @param adjacency_matrix
 * @param qubit_list
 */
void Topology::floyd_warshall(std::vector<std::vector<QubitIdType>>& adjacency_matrix, const std::vector<PhysicalQubit>& qubit_list) {
    _set_weight(adjacency_matrix, qubit_list);
    _init_predecessor_and_distance(adjacency_matrix, qubit_list);
    for (size_t k = 0; k < _num_qubit; k++) {
        spdlog::debug("Including vertex({}):", k);
        for (size_t i = 0; i < _num_qubit; i++) {
            for (size_t j = 0; j < _num_qubit; j++) {
                if ((_distance[i][j] > _distance[i][k] + _distance[k][j]) && (_distance[i][k] != _max_dist)) {
                    _distance[i][j]    = _distance[i][k] + _distance[k][j];
                    _predecessor[i][j] = _predecessor[k][j];
                }
            }
        }

        spdlog::debug("Predecessor Matrix:");
        for (auto& row : _predecessor) {
            spdlog::debug("{:5}", fmt::join(
                                      row | std::views::transform([](auto j) { return (j == max_qubit_id) ? std::string{"/"} : std::to_string(j); }), ""));
        }
        spdlog::debug("Distance Matrix:");
        for (auto& row : _distance) {
            spdlog::debug("{:5}", fmt::join(
                                      row | std::views::transform([this](size_t j) { return (j == _max_dist) ? std::string{"X"} : std::to_string(j); }), ""));
        }
    }
}

/**
 * @brief Print information of the edge (a,b)
 *
 * @param a Index of first qubit
 * @param b Index of second qubit
 */
void Topology::print_single_edge(size_t a, size_t b) const {
    auto query = (a < b) ? std::make_pair(a, b) : std::make_pair(b, a);
    if (_adjacency_info.contains(query)) {
        fmt::println("({:>3}, {:>3})    Delay: {:>8.3f}    Error: {:>8.5f}", a, b, _adjacency_info.at(query)._time, _adjacency_info.at(query)._error);
    } else {
        fmt::println("No connection between {:>3} and {:>3}.", a, b);
    }
}

// SECTION - Class PhysicalQubit Member Functions

/**
 * @brief Operator overloading
 *
 * @param os
 * @param q
 * @return ostream&
 */
std::ostream& operator<<(std::ostream& os, PhysicalQubit const& q) {
    return os << fmt::format("{}", q);
}

/**
 * @brief Mark qubit
 *
 * @param source false: from 0, true: from 1
 * @param pred predecessor
 */
void PhysicalQubit::mark(bool source, QubitIdType pred) {
    _marked = true;
    _source = source;
    _pred   = pred;
}

/**
 * @brief Take the route
 *
 * @param cost
 * @param swapTime
 */
void PhysicalQubit::take_route(size_t cost, size_t swap_time) {
    _cost      = cost;
    _swap_time = swap_time;
    _taken     = true;
}

/**
 * @brief Reset qubit
 *
 */
void PhysicalQubit::reset() {
    _marked = false;
    _taken  = false;
    _cost   = _occupied_time;
}

// SECTION - Class Device Member Functions

/**
 * @brief Get next swap cost
 *
 * @param source
 * @param target
 * @return tuple<size_t, size_t> (index of next qubit, cost)
 */
std::tuple<QubitIdType, QubitIdType> Device::get_next_swap_cost(QubitIdType source, QubitIdType target) {
    auto const next_idx  = _topology->get_predecessor(target, source);
    auto const& q_source = get_physical_qubit(source);
    auto const& q_next   = get_physical_qubit(next_idx);
    auto const cost      = std::max(q_source.get_occupied_time(), q_next.get_occupied_time());

    assert(q_source.is_adjacency(q_next));
    return {next_idx, cost};
}

/**
 * @brief Get physical qubit id by logical id
 *
 * @param id logical
 * @return size_t
 */
QubitIdType Device::get_physical_by_logical(QubitIdType id) {
    for (auto& phy : _qubit_list) {
        if (phy.get_logical_qubit() == id) {
            return phy.get_id();
        }
    }
    return max_qubit_id;
}

/**
 * @brief Add adjacency pair (a,b)
 *
 * @param a Id of first qubit
 * @param b Id of second qubit
 */
void Device::add_adjacency(QubitIdType a, QubitIdType b) {
    if (a > b) std::swap(a, b);
    _qubit_list[a].add_adjacency(_qubit_list[b].get_id());
    _qubit_list[b].add_adjacency(_qubit_list[a].get_id());
    constexpr DeviceInfo default_info = {._time = 0.0, ._error = 0.0};
    _topology->add_adjacency_info(a, b, default_info);
}

/**
 * @brief Apply gate to device
 *
 * @param op
 */
void Device::apply_gate(qcir::QCirGate const& op, size_t time_begin) {
    auto qubits = op.get_qubits();
    auto& q0    = get_physical_qubit(qubits[0]);
    auto& q1    = get_physical_qubit(qubits[1]);

    if (op.get_operation() == SwapGate{}) {
        auto temp = q0.get_logical_qubit();
        q0.set_logical_qubit(q1.get_logical_qubit());
        q1.set_logical_qubit(temp);
        q0.set_occupied_time(time_begin + get_delay(op));
        q1.set_occupied_time(time_begin + get_delay(op));
    } else if (op.get_num_qubits() == 2) {
        q0.set_occupied_time(time_begin + get_delay(op));
        q1.set_occupied_time(time_begin + get_delay(op));
    } else {
        DVLAB_ASSERT(false, fmt::format("Unknown gate type ({}) at apply_gate()!!", op.get_operation().get_repr()));
    }
}

/**
 * @brief get mapping of physical qubit
 *
 * @return vector<size_t> (index of physical qubit)
 */
std::vector<std::optional<size_t>> Device::mapping() const {
    std::vector<std::optional<size_t>> ret;
    ret.resize(_qubit_list.size());
    for (auto const& [id, qubit] : tl::views::enumerate(_qubit_list)) {
        ret[id] = qubit.get_logical_qubit();
    }
    return ret;
}

/**
 * @brief Place logical qubit
 *
 * @param assign
 */
void Device::place(std::vector<QubitIdType> const& assignment) {
    for (size_t i = 0; i < assignment.size(); ++i) {
        assert(_qubit_list[assignment[i]].get_logical_qubit() == std::nullopt);
        _qubit_list[assignment[i]].set_logical_qubit(i);
    }
}

/**
 * @brief Calculate Shortest Path
 *
 */
void Device::calculate_path() {
    _topology->clear_predecessor();
    _topology->clear_distance();

    std::vector<std::vector<QubitIdType>> adjacency_matrix;
    adjacency_matrix.resize(get_num_qubits());

    for (size_t i = 0; i < get_num_qubits(); i++) {
        adjacency_matrix[i].resize(get_num_qubits(), default_max_dist);
        for (size_t j = 0; j < get_num_qubits(); j++) {
            if (i == j)
                adjacency_matrix[i][j] = 0;
        }
    }
    _topology->floyd_warshall(adjacency_matrix, _qubit_list);
}

/**
 * @brief Get shortest path from `s` to `t`
 *
 * @param s start
 * @param t terminate
 * @return vector<PhyQubit>&
 */
std::vector<PhysicalQubit> Device::get_path(QubitIdType src, QubitIdType dest) const {
    std::vector<PhysicalQubit> path;
    path.emplace_back(_qubit_list.at(src));
    if (src == dest) return path;
    auto new_pred = _topology->get_predecessor(dest, src);
    path.emplace_back(new_pred);
    while (true) {
        new_pred = _topology->get_predecessor(dest, new_pred);
        if (new_pred == max_qubit_id) break;
        path.emplace_back(_qubit_list.at(new_pred));
    }
    return path;
}

/**
 * @brief Read Device
 *
 * @param filename
 * @return true
 * @return false
 */
bool Device::read_device(std::string const& filename) {
    std::ifstream topo_file(filename);
    if (!topo_file.is_open()) {
        spdlog::error("Cannot open the file \"{}\"!!", filename);
        return false;
    }
    std::string str = "", token = "", data = "";

    // NOTE - Device name
    while (str.empty()) {
        std::getline(topo_file, str);
        str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
    }
    size_t token_end = dvlab::str::str_get_token(str, token, 0, ": ");
    data             = str.substr(token_end + 1);

    _topology->set_name(std::string{dvlab::str::trim_spaces(data)});

    // NOTE - Qubit num
    str = "", token = "", data = "";
    while (str.empty()) {
        std::getline(topo_file, str);
        str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
    }
    token_end = dvlab::str::str_get_token(str, token, 0, ": ");
    data      = str.substr(token_end + 1);
    data      = dvlab::str::trim_spaces(data);
    auto qbn  = dvlab::str::from_string<unsigned>(data);
    if (!qbn.has_value()) {
        spdlog::error("The number of qubit is not a positive integer!!");
        return false;
    }
    _num_qubit = qbn.value();
    _topology->set_num_qubits(_num_qubit);
    // NOTE - Gate set
    str = "", token = "", data = "";
    while (str.empty()) {
        std::getline(topo_file, str);
        str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
    }
    if (!_parse_gate_set(str)) return false;

    // NOTE - Coupling map
    str = "", token = "", data = "";
    while (str.empty()) {
        std::getline(topo_file, str);
        str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
    }

    token_end = dvlab::str::str_get_token(str, token, 0, ": ");
    data      = str.substr(token_end + 1);
    data      = dvlab::str::trim_spaces(data);
    data      = dvlab::str::remove_brackets(data, '[', ']');
    std::vector<std::vector<float>> cx_err, cx_delay;
    std::vector<std::vector<size_t>> adj_list;
    std::vector<float> sg_err, sg_delay;
    if (!_parse_size_t_pairs(data, adj_list))
        return false;

    // NOTE - Parse Information
    if (!_parse_info(topo_file, cx_err, cx_delay, sg_err, sg_delay)) return false;

    // NOTE - Finish parsing, store the topology
    _qubit_list.reserve(adj_list.size());

    for (size_t i = 0; i < adj_list.size(); ++i) {
        _qubit_list.emplace_back(PhysicalQubit(i));
    }

    for (size_t i = 0; i < adj_list.size(); i++) {
        for (size_t j = 0; j < adj_list[i].size(); j++) {
            if (adj_list[i][j] > i) {
                add_adjacency(i, adj_list[i][j]);
                _topology->add_adjacency_info(i, adj_list[i][j], {._time = cx_delay[i][j], ._error = cx_err[i][j]});
            }
        }
    }

    assert(sg_err.size() == sg_delay.size());
    for (size_t i = 0; i < sg_err.size(); i++) {
        _topology->add_qubit_info(i, {._time = sg_delay[i], ._error = sg_err[i]});
    }

    calculate_path();
    return true;
}

/**
 * @brief Parse gate set
 *
 * @param str
 * @return true
 * @return false
 */
bool Device::_parse_gate_set(std::string const& gate_set_str) {
    std::string _;
    auto const token_end = dvlab::str::str_get_token(gate_set_str, _, 0, ": ");
    auto data            = gate_set_str.substr(token_end + 1);
    data                 = dvlab::str::trim_spaces(data);
    data                 = dvlab::str::remove_brackets(data, '{', '}');
    auto gate_set_view =
        dvlab::str::views::tokenize(data, ',') |
        std::views::transform([](auto const& str) { return dvlab::str::tolower_string(str); }) |
        std::views::transform([&](auto const& str) -> std::optional<std::string> {
            if (auto op = qcir::str_to_operation(str); op.has_value()) {
                _topology->add_gate_type(op->get_repr().substr(0, op->get_repr().find_first_of('(')));
                return std::make_optional(op->get_type());
            }
            if (auto op = qcir::str_to_operation(str, {dvlab::Phase()}); op.has_value()) {
                _topology->add_gate_type(op->get_repr().substr(0, op->get_repr().find_first_of('(')));
                return std::make_optional(op->get_type());
            }
            return std::nullopt;
        });

    return std::ranges::all_of(gate_set_view, [](auto const& gate_type) { return gate_type.has_value(); });
}

/**
 * @brief Parse device information including SGERROR, SGTIME, CNOTERROR, and CNOTTIME
 *
 * @param f
 * @param cxErr
 * @param cxDelay
 * @param sgErr
 * @param sgDelay
 * @return true
 * @return false
 */
bool Device::_parse_info(std::ifstream& f, std::vector<std::vector<float>>& cx_error, std::vector<std::vector<float>>& cx_delay, std::vector<float>& single_error, std::vector<float>& single_delay) {
    std::string str = "", token = "";
    while (true) {
        while (str.empty()) {
            if (f.eof()) break;
            std::getline(f, str);
            str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
        }
        auto const token_end = dvlab::str::str_get_token(str, token, 0, ": ");
        auto const data      = dvlab::str::trim_spaces(str.substr(token_end + 1));

        if (token == "SGERROR") {
            if (!_parse_singles(std::string{data}, single_error)) return false;
        } else if (token == "SGTIME") {
            if (!_parse_singles(std::string{data}, single_delay)) return false;
        } else if (token == "CNOTERROR") {
            if (!_parse_float_pairs(std::string{data}, cx_error)) return false;
        } else if (token == "CNOTTIME") {
            if (!_parse_float_pairs(std::string{data}, cx_delay)) return false;
        }
        if (f.eof()) {
            break;
        }
        std::getline(f, str);
        str = dvlab::str::trim_spaces(dvlab::str::trim_comments(str));
    }

    return true;
}

/**
 * @brief Parse device qubits information
 *
 * @param data
 * @param container
 * @return true
 * @return false
 */
bool Device::_parse_singles(std::string const& data, std::vector<float>& container) {
    std::string const buffer = dvlab::str::remove_brackets(data, '[', ']');

    for (auto const& token : dvlab::str::views::tokenize(buffer, ',')) {
        auto fl = dvlab::str::from_string<float>(dvlab::str::trim_spaces(token));
        if (!fl.has_value()) {
            spdlog::error("The number `{}` is not a float!!", token);
            return false;
        }
        container.emplace_back(fl.value());
    }
    return true;
}

/**
 * @brief Parse device edges information with type is float
 *
 * @param data
 * @param container
 * @return true
 * @return false
 */
bool Device::_parse_float_pairs(std::string const& data, std::vector<std::vector<float>>& containers) {
    for (auto const& outer_token : dvlab::str::views::tokenize(data, '[')) {
        std::string const buffer{outer_token.substr(0, outer_token.find_first_of(']'))};
        auto floats =
            dvlab::str::views::tokenize(buffer, ',') |
            std::views::transform([](auto const& str) {
                auto result = dvlab::str::from_string<float>(str);
                if (!result.has_value()) {
                    spdlog::error("The number `{}` is not a float!!", str);
                    return std::optional<float>{};
                }
                return result;
            });

        if (std::ranges::any_of(floats, [](auto const& fl) { return !fl.has_value(); })) {
            return false;
        }

        containers.emplace_back(floats | std::views::transform([](auto const& fl) { return fl.value(); }) | tl::to<std::vector>());
    }
    return true;
}

/**
 * @brief Parse device edges information with type is size_t
 *
 * @param data
 * @param container
 * @return true
 * @return false
 */
bool Device::_parse_size_t_pairs(std::string const& data, std::vector<std::vector<size_t>>& containers) {
    for (auto const& outer_token : dvlab::str::views::tokenize(data, '[')) {
        std::string const buffer{outer_token.substr(0, outer_token.find_first_of(']'))};
        auto qubit_ids =
            dvlab::str::views::tokenize(buffer, ',') |
            std::views::transform([](auto const& str) {
                auto result = dvlab::str::from_string<size_t>(str);
                if (!result.has_value()) {
                    spdlog::error("The number `{}` is not a positive integer!!", str);
                    return std::optional<size_t>{};
                }
                return result;
            });

        if (std::ranges::any_of(qubit_ids, [](auto const& fl) { return !fl.has_value(); })) {
            return false;
        }

        containers.emplace_back(qubit_ids | std::views::transform([](auto const& fl) { return fl.value(); }) | tl::to<std::vector>());
    }

    return true;
}

/**
 * @brief Print physical qubits and their adjacencies
 *
 * @param cand a vector of qubits to be printed
 */
void Device::print_qubits(std::vector<size_t> candidates) const {
    for (auto& c : candidates) {
        if (c >= _num_qubit) {
            spdlog::error("Error: the maximum qubit id is {}!!", _num_qubit - 1);
            return;
        }
    }
    fmt::println("");
    std::vector<PhysicalQubit> qubits;
    qubits.resize(_num_qubit);
    for (auto const& [idx, info] : tl::views::enumerate(_qubit_list)) {
        qubits[idx] = info;
    }
    if (candidates.empty()) {
        for (size_t i = 0; i < qubits.size(); i++) {
            fmt::println("ID: {:>3}    {}Adjs: {:>3}", i, _topology->get_qubit_info(i), fmt::join(qubits[i].get_adjacencies(), " "));
        }
        fmt::println("Total #Qubits: {}", _num_qubit);
    } else {
        std::ranges::sort(candidates);
        for (auto& p : candidates) {
            fmt::println("ID: {:>3}    {}Adjs: {:>3}", p, _topology->get_qubit_info(p), fmt::join(qubits[p].get_adjacencies(), " "));
        }
    }
}

/**
 * @brief Print device edge
 *
 * @param cand Empty: print all. Single element [a]: print edges connecting to a. Two elements [a,b]: print edge (a,b).
 */
void Device::print_edges(std::vector<size_t> candidates) const {
    for (auto& c : candidates) {
        if (c >= _num_qubit) {
            spdlog::error("the maximum qubit id is {}!!", _num_qubit - 1);
            return;
        }
    }
    fmt::println("");
    std::vector<PhysicalQubit> qubits;
    qubits.resize(_num_qubit);
    for (auto const& [idx, info] : tl::views::enumerate(_qubit_list)) {
        qubits[idx] = info;
    }
    if (candidates.empty()) {
        size_t cnt = 0;
        for (size_t i = 0; i < _num_qubit; i++) {
            for (auto& q : qubits[i].get_adjacencies()) {
                if (std::cmp_less(i, q)) {
                    cnt++;
                    _topology->print_single_edge(i, q);
                }
            }
        }
        assert(cnt == _topology->get_num_adjacencies());
        fmt::println("Total #Edges: {}", cnt);
    } else if (candidates.size() == 1) {
        for (auto& q : qubits[candidates[0]].get_adjacencies()) {
            _topology->print_single_edge(candidates[0], q);
        }
        fmt::println("Total #Edges: {}", qubits[candidates[0]].get_adjacencies().size());
    } else if (candidates.size() == 2) {
        _topology->print_single_edge(candidates[0], candidates[1]);
    }
}

/**
 * @brief Print information of Topology
 *
 */
void Device::print_topology() const {
    fmt::println("Topology: {} ({} qubits, {} edges)", get_name(), _qubit_list.size(), _topology->get_num_adjacencies());
    auto const tmp = _topology->get_gate_set();  // circumvents g++ 11.4 compiler bug
    fmt::println("Gate Set: {}", fmt::join(tmp | std::views::transform([](std::string const& gtype) { return dvlab::str::toupper_string(gtype); }), ", "));
}

/**
 * @brief Print shortest path from `s` to `t`
 *
 * @param s start
 * @param t terminate
 */
void Device::print_path(QubitIdType src, QubitIdType dest) const {
    fmt::println("");
    for (auto& c : {src, dest}) {
        if (std::cmp_greater_equal(c, _num_qubit)) {
            spdlog::error("the maximum qubit id is {}!!", _num_qubit - 1);
            return;
        }
    }
    std::vector<PhysicalQubit> const& path = get_path(src, dest);
    if (path.front().get_id() != src && path.back().get_id() != dest)
        fmt::println("No path between {} and {}", src, dest);
    else {
        fmt::println("Path from {} to {}:", src, dest);
        size_t cnt = 0;
        for (auto& v : path) {
            constexpr size_t num_cols = 10;
            fmt::print("{:4} ", v.get_id());
            if (++cnt % num_cols == 0) fmt::println("");
        }
    }
}

/**
 * @brief Print Mapping (Physical : Logical)
 *
 */
void Device::print_mapping() {
    fmt::println("----------Mapping---------");
    for (size_t i = 0; i < _num_qubit; i++) {
        fmt::println("{:<5} : {}", i, _qubit_list[i].get_logical_qubit());
    }
}

/**
 * @brief Print device status
 *
 */
void Device::print_status() const {
    fmt::println("Device Status:");
    std::vector<PhysicalQubit> qubits;
    qubits.resize(_num_qubit);
    for (auto const& [idx, info] : tl::views::enumerate(_qubit_list)) {
        qubits[idx] = info;
    }
    for (size_t i = 0; i < qubits.size(); ++i) {
        fmt::println("{}", qubits[i]);
    }
    fmt::println("");
}

}  // namespace qsyn::device
