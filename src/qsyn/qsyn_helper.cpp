/****************************************************************************
  PackageName  [ qsyn ]
  Synopsis     [ Define qsynrc functions ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/

#include "./qsyn_helper.hpp"

#include <fmt/chrono.h>
#include <fmt/std.h>
#include <spdlog/spdlog.h>

#include <csignal>

#include "argparse/arg_parser.hpp"
#include "argparse/arg_type.hpp"
#include "cli/cli.hpp"
#include "cmd/conversion_cmd.hpp"
#include "cmd/device_cmd.hpp"
#include "cmd/duostra_cmd.hpp"
#include "cmd/extractor_cmd.hpp"
#include "cmd/qcir_cmd.hpp"
#include "cmd/tableau_cmd.hpp"
#include "cmd/tensor_cmd.hpp"
#include "cmd/zx_cmd.hpp"
#include "util/sysdep.hpp"

namespace qsyn {

namespace {

std::filesystem::path const default_qsynrc_path = std::invoke([]() {
    auto const home_dir = dvlab::utils::get_home_directory();
    if (!home_dir) {
        spdlog::critical("Cannot find home directory");
        std::exit(1);
    }
    return std::filesystem::path{home_dir.value()} / ".config/qsyn/qsynrc";
});

void create_default_qsynrc(dvlab::CommandLineInterface& cli, std::filesystem::path const& qsynrc_path) {
    namespace fs = std::filesystem;

    if (!fs::is_directory(qsynrc_path.parent_path()) && !fs::create_directories(qsynrc_path.parent_path())) {
        spdlog::critical("Cannot create directory {}", qsynrc_path.parent_path());
        return;
    }
    // clang-format off
    std::ofstream{qsynrc_path} <<
    // embedded default qsynrc file
        #include "./qsynrc.default"
    ;
    // clang-format on
    cli.source_dofile(qsynrc_path, {}, false);
}

dvlab::Command create_qsynrc_cmd(dvlab::CommandLineInterface& cli) {
    using namespace dvlab::argparse;
    return dvlab::Command{
        "create-qsynrc",
        [](ArgumentParser& parser) {
            parser.description(fmt::format("Create a default qsynrc file at {}", default_qsynrc_path));
            parser.add_argument<bool>("-r", "--replace")
                .action(store_true)
                .help("force-replace the existing qsynrc file");
            return parser;
        },
        [&](ArgumentParser const& parser) {
            namespace fs        = std::filesystem;
            auto const home_dir = dvlab::utils::get_home_directory();
            if (!home_dir) {
                spdlog::critical("Cannot find home directory");
                return dvlab::CmdExecResult::error;
            }

            if (fs::exists(default_qsynrc_path)) {
                if (parser.get<bool>("--replace")) {
                    fmt::println("Replacing qsynrc at {}", default_qsynrc_path);
                } else {
                    fmt::println("qsynrc already exists at {}. Specify `-r` flag to replace it.", default_qsynrc_path);
                    return dvlab::CmdExecResult::error;
                }
            }

            create_default_qsynrc(cli, default_qsynrc_path);

            return dvlab::CmdExecResult::done;
        }};
}

bool add_qsyn_cmds(dvlab::CommandLineInterface& cli) {
    return cli.add_command(create_qsynrc_cmd(cli));
}

}  // namespace

bool read_qsynrc_file(dvlab::CommandLineInterface& cli, std::filesystem::path qsynrc_path) {
    namespace fs = std::filesystem;
    if (qsynrc_path.empty()) {
        auto const home_dir = dvlab::utils::get_home_directory();
        if (!home_dir) {
            spdlog::critical("Cannot find home directory");
            return false;
        }
        qsynrc_path = default_qsynrc_path;
        if (!fs::exists(qsynrc_path)) {
            create_default_qsynrc(cli, qsynrc_path);
            cli.clear_history();
            return true;
        }
    }

    auto const result = cli.source_dofile(qsynrc_path, {}, false);
    cli.clear_history();

    if (result == dvlab::CmdExecResult::error) {
        spdlog::critical("Some errors occurred while reading the qsynrc file from {}", qsynrc_path);
        return false;
    }

    return true;
}

bool initialize_qsyn(
    dvlab::CommandLineInterface& cli, qsyn::device::DeviceMgr& device_mgr, qsyn::qcir::QCirMgr& qcir_mgr,
    qsyn::tensor::TensorMgr& tensor_mgr, qsyn::zx::ZXGraphMgr& zxgraph_mgr, qsyn::experimental::TableauMgr& tableau_mgr) {
    spdlog::set_pattern("%L%v");
    spdlog::set_level(spdlog::level::warn);

    return dvlab::add_cli_common_cmds(cli) &&
           qsyn::add_qsyn_cmds(cli) &&
           qsyn::device::add_device_cmds(cli, device_mgr) &&
           qsyn::duostra::add_duostra_cmds(cli, qcir_mgr, device_mgr) &&
           qsyn::add_conversion_cmds(cli, qcir_mgr, tensor_mgr, zxgraph_mgr, tableau_mgr) &&
           qsyn::extractor::add_extract_cmds(cli, zxgraph_mgr, qcir_mgr) &&
           qsyn::qcir::add_qcir_cmds(cli, qcir_mgr) &&
           qsyn::tensor::add_tensor_cmds(cli, tensor_mgr) &&
           qsyn::zx::add_zx_cmds(cli, zxgraph_mgr) &&
           qsyn::experimental::add_tableau_command(cli, tableau_mgr);
}

dvlab::argparse::ArgumentParser get_qsyn_parser(std::string_view const prog_name) {
    using namespace dvlab::argparse;

    auto parser = ArgumentParser(std::string{prog_name}, {.add_help_action    = true,
                                                          .add_version_action = true,
                                                          .exit_on_failure    = true,
                                                          .version            = QSYN_VERSION});
    auto mutex  = parser.add_mutually_exclusive_group();

    mutex.add_argument<std::string>("filepath")
        .usage(fmt::format("{} {}{}{}...",
                           dvlab::fmt_ext::styled_if_ansi_supported("filepath", fmt::emphasis::bold),
                           dvlab::fmt_ext::styled_if_ansi_supported("[", fmt::fg(fmt::terminal_color::yellow)),
                           dvlab::fmt_ext::styled_if_ansi_supported("arg", fmt::emphasis::bold),
                           dvlab::fmt_ext::styled_if_ansi_supported("]", fmt::fg(fmt::terminal_color::yellow))))
        .nargs(NArgsOption::zero_or_more)
        .help("run the script file at the specified path. Arguments can be passed after the filepath.");

    mutex.add_argument<std::string>("-c", "--command")
        .help("run the command passed as a string");

    parser.add_argument<bool>("-q", "--quiet")
        .action(store_true)
        .help("suppress echoing of commands when supplying commands from `-c` or `-f` flags. This argument also implies `--no-version`");

    parser.add_argument<bool>("--no-version")
        .action(store_true)
        .help("suppress version information on start up");

    parser.add_argument<std::string>("--qsynrc-path")
        .default_value("")
        .help("specify the path to the qsynrc file");

    parser.add_argument<bool>("-f", "--file")
        .action(store_true)
        .help("This flag is deprecated but still supported for backward compatibility. To run a script file with commands printing, use `-v` flag with a filepath.");

    parser.add_argument<bool>("-v", "--verbose")
        .action(store_true)
        .help("print the commands before executing them; only valid when `-c` or a filepath is specified");

    return parser;
}

}  // namespace qsyn
