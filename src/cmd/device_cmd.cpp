/****************************************************************************
  PackageName  [ device ]
  Synopsis     [ Define device package commands ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./device_cmd.hpp"

#include <spdlog/spdlog.h>

#include <memory>
#include <string>

#include "./device_mgr.hpp"
#include "device/device.hpp"
#include "qsyn/qsyn_type.hpp"
#include "util/data_structure_manager_common_cmd.hpp"

using namespace dvlab::argparse;
using dvlab::CmdExecResult;

namespace qsyn::device {

std::function<bool(size_t const&)> valid_device_id(qsyn::device::DeviceMgr const& device_mgr) {
    return [&device_mgr](size_t const& id) {
        if (device_mgr.is_id(id)) return true;
        spdlog::error("Device {} does not exist!!", id);
        return false;
    };
};

dvlab::Command device_checkout_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"checkout",
            [&device_mgr](ArgumentParser& parser) {
                parser.description("checkout to Device <id> in DeviceMgr");

                parser.add_argument<size_t>("id")
                    .constraint(valid_device_id(device_mgr))
                    .help("the ID of the device");
            },
            [&device_mgr](ArgumentParser const& parser) {
                device_mgr.checkout(parser.get<size_t>("id"));
                return CmdExecResult::done;
            }};
}

dvlab::Command device_clear_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"clear",
            [](ArgumentParser& parser) {
                parser.description("clear DeviceMgr");
            },
            [&device_mgr](ArgumentParser const& /*parser*/) {
                device_mgr.clear();
                return CmdExecResult::done;
            }};
}

dvlab::Command device_delete_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"delete",
            [&device_mgr](ArgumentParser& parser) {
                parser.description("remove a Device from DeviceMgr");

                parser.add_argument<size_t>("id")
                    .constraint(valid_device_id(device_mgr))
                    .help("the ID of the device");
            },
            [&device_mgr](ArgumentParser const& parser) {
                device_mgr.remove(parser.get<size_t>("id"));
                return CmdExecResult::done;
            }};
}

dvlab::Command device_read_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"read",
            [](ArgumentParser& parser) {
                parser.description("read a device topology");

                parser.add_argument<std::string>("filepath")
                    .help("the filepath to device file");

                parser.add_argument<bool>("-r", "--replace")
                    .action(store_true)
                    .help("if specified, replace the current device; otherwise store to a new one");
            },
            [&device_mgr](ArgumentParser const& parser) {
                qsyn::device::Device buffer_device;
                auto filepath = parser.get<std::string>("filepath");
                auto replace  = parser.get<bool>("--replace");

                if (!buffer_device.read_device(filepath)) {
                    spdlog::error("the format in \"{}\" has something wrong!!", filepath);
                    return CmdExecResult::error;
                }

                if (device_mgr.empty() || !replace) {
                    device_mgr.add(device_mgr.get_next_id(), std::make_unique<qsyn::device::Device>(std::move(buffer_device)));
                } else {
                    device_mgr.set(std::make_unique<qsyn::device::Device>(std::move(buffer_device)));
                }

                return CmdExecResult::done;
            }};
}

dvlab::Command device_list_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"list",
            [](ArgumentParser& parser) {
                parser.description("list info about Devices");
            },
            [&device_mgr](ArgumentParser const& /* parser */) {
                device_mgr.print_list();

                return CmdExecResult::done;
            }};
}

dvlab::Command device_print_cmd(qsyn::device::DeviceMgr& device_mgr) {
    return {"print",
            [](ArgumentParser& parser) {
                parser.description("print info of device topology");

                auto mutex = parser.add_mutually_exclusive_group().required(false);

                mutex.add_argument<size_t>("-e", "--edges")
                    .nargs(0, 2)
                    .help(
                        "print information of edges. "
                        "If no qubit ID is specified, print for all edges; "
                        "if one qubit ID specified, list the adjacent edges to the qubit; "
                        "if two qubit IDs are specified, list the edge between them");

                mutex.add_argument<size_t>("-q", "--qubits")
                    .nargs(NArgsOption::zero_or_more)
                    .help(
                        "print information of qubits. "
                        "If no qubit ID is specified, print for all qubits;"
                        "otherwise, print information of the specified qubit IDs");

                mutex.add_argument<QubitIdType>("-p", "--path")
                    .nargs(2)
                    .metavar("(q1, q2)")
                    .help(
                        "print routing paths between q1 and q2");
            },
            [&device_mgr](ArgumentParser const& parser) {
                if (!dvlab::utils::mgr_has_data(device_mgr)) return CmdExecResult::error;

                if (parser.parsed("--edges")) {
                    device_mgr.get()->print_edges(parser.get<std::vector<size_t>>("--edges"));
                    return CmdExecResult::done;
                }
                if (parser.parsed("--qubits")) {
                    device_mgr.get()->print_qubits(parser.get<std::vector<size_t>>("--qubits"));
                    return CmdExecResult::done;
                }
                if (parser.parsed("--path")) {
                    auto qids = parser.get<std::vector<QubitIdType>>("--path");
                    device_mgr.get()->print_path(qids[0], qids[1]);
                    return CmdExecResult::done;
                }

                device_mgr.get()->print_topology();
                return CmdExecResult::done;
            }};
}

dvlab::Command device_cmd(qsyn::device::DeviceMgr& device_mgr) {
    auto cmd = dvlab::utils::mgr_root_cmd(device_mgr);
    // print functions
    cmd.add_subcommand("device-cmd-group", dvlab::utils::mgr_list_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_print_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_checkout_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", device_read_cmd(device_mgr));
    cmd.add_subcommand("device-cmd-group", dvlab::utils::mgr_delete_cmd(device_mgr));
    return cmd;
}

bool add_device_cmds(dvlab::CommandLineInterface& cli, qsyn::device::DeviceMgr& device_mgr) {
    if (!cli.add_command(device_cmd(device_mgr))) {
        spdlog::critical("Registering \"device\" commands fails... exiting");
        return false;
    }
    return true;
}

}  // namespace qsyn::device
