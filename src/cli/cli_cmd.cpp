/****************************************************************************
  PackageName  [ cli ]
  Synopsis     [ Define common commands for any CLI ]
  Author       [ Design Verification Lab ]
  Copyright    [ Copyright(c) 2023 DVLab, GIEE, NTU, Taiwan ]
****************************************************************************/
#include <spdlog/spdlog.h>

#include "cli/cli.hpp"
#include "spdlog/common.h"
#include "util/cin_cout_cerr.hpp"
#include "util/dvlab_string.hpp"
#include "util/sysdep.hpp"
#include "util/usage.hpp"

namespace dvlab {

using namespace dvlab::argparse;

Command alias_cmd(CommandLineInterface& cli) {
    return {
        "alias",
        [](ArgumentParser& parser) {
            parser.description("alias command string to another name");

            parser.add_argument<std::string>("alias")
                .required(false)
                .help("the alias to add");

            parser.add_argument<std::string>("replace-str")
                .required(false)
                .help("the string to alias to");

            parser.add_argument<std::string>("-d", "--delete")
                .metavar("alias")
                .help("delete the alias");

            parser.add_argument<bool>("-l", "--list")
                .action(store_true)
                .help("list all aliases");
        },

        [&cli](ArgumentParser const& parser) {
            if (parser.parsed("-l")) {
                cli.list_all_aliases();
                return CmdExecResult::done;
            }
            if (parser.parsed("-d")) {
                if (parser.parsed("alias") || parser.parsed("replace-str")) {
                    fmt::println(stderr, "Error: cannot specify replacement string when deleting alias!!");
                    return CmdExecResult::error;
                }
                if (!cli.remove_alias(parser.get<std::string>("-d"))) {
                    return CmdExecResult::error;
                }
                return CmdExecResult::done;
            }

            if (!(parser.parsed("alias") && parser.parsed("replace-str"))) {
                fmt::println(stderr, "Error: alias and replacement string must be specified!!");
                return CmdExecResult::error;
            }

            auto alias       = parser.get<std::string>("alias");
            auto replace_str = parser.get<std::string>("replace-str");

            if (cli.add_alias(alias, replace_str)) {
                return CmdExecResult::error;
            }

            return CmdExecResult::done;
        }};
}

Command echo_cmd() {
    return {
        "echo",
        [](ArgumentParser& parser) {
            parser.description("print the string to the standard output");

            parser.add_argument<std::string>("message")
                .nargs(NArgsOption::zero_or_more)
                .help("the message to print");
        },
        [](ArgumentParser const& parser) {
            fmt::println("{}", fmt::join(parser.get<std::vector<std::string>>("message"), " "));
            return CmdExecResult::done;
        }};
}

Command set_variable_cmd(CommandLineInterface& cli) {
    return {
        "set",
        [](ArgumentParser& parser) {
            parser.description("set a variable");

            parser.add_argument<std::string>("variable")
                .required(false)
                .help("the variable to set");

            parser.add_argument<std::string>("value")
                .required(false)
                .help("the value to set");

            parser.add_argument<std::string>("-d", "--delete")
                .metavar("variable")
                .help("delete the variable");

            parser.add_argument<bool>("-l", "--list")
                .action(store_true)
                .help("list all variables");
        },

        [&cli](ArgumentParser const& parser) {
            if (parser.parsed("-l")) {
                cli.list_all_variables();
                return CmdExecResult::done;
            }

            if (parser.parsed("-d")) {
                if (parser.parsed("variable") || parser.parsed("value")) {
                    fmt::println(stderr, "Error: cannot specify values when deleting variable!!");
                    return CmdExecResult::error;
                }
                if (!cli.remove_variable(parser.get<std::string>("-d"))) {
                    return CmdExecResult::error;
                }
                return CmdExecResult::done;
            }

            if (!(parser.parsed("variable") && parser.parsed("value"))) {
                fmt::println(stderr, "Error: variable and value must be specified!!");
                return CmdExecResult::error;
            }

            auto variable = parser.get<std::string>("variable");
            auto value    = parser.get<std::string>("value");

            if (std::ranges::any_of(variable, [](char ch) { return isspace(ch); })) {
                fmt::println(stderr, "Error: variable cannot contain whitespaces!!");
                return CmdExecResult::error;
            }

            if (!cli.add_variable(variable, value)) {
                return CmdExecResult::error;
            }

            return CmdExecResult::done;
        }};
}

Command help_cmd(CommandLineInterface& cli) {
    return {
        "help",
        [](ArgumentParser& parser) {
            parser.description("shows helping message to commands");

            parser.add_argument<std::string>("command")
                .default_value("")
                .nargs(NArgsOption::optional)
                .help("if specified, display help message to a command");
        },

        [&cli](ArgumentParser const& parser) {
            auto command = parser.get<std::string>("command");
            if (command.empty()) {
                cli.list_all_commands();
                return CmdExecResult::done;
            }

            // this also handles the case when `command` is an alias to itself

            auto alias_replacement_string = cli.get_alias_replacement_string(command);
            if (alias_replacement_string.has_value()) {
                fmt::println("`{}` is an alias to `{}`.", command, alias_replacement_string.value());
                fmt::println("Showing help message to `{}`:\n", alias_replacement_string.value());
                command = cli.get_first_token(alias_replacement_string.value());
            }

            if (auto e = cli.get_command(command)) {
                e->print_help();
                return CmdExecResult::done;
            }

            fmt::println(stderr, "Error: illegal command or alias!! ({})", command);
            return CmdExecResult::error;
        }};
}

Command quit_cmd(CommandLineInterface& cli) {
    return {"quit",
            [](ArgumentParser& parser) {
                parser.description("quit qsyn");

                parser.add_argument<bool>("-f", "--force")
                    .action(store_true)
                    .help("quit without reaffirming");
            },
            [&cli](ArgumentParser const& parser) {
                using namespace std::string_view_literals;
                if (parser.get<bool>("--force")) return CmdExecResult::quit;

                std::string const prompt = "Are you sure you want to exit (Yes/[No])? ";

                auto const [exec_result, input] = cli.listen_to_input(std::cin, prompt, {.allow_browse_history = false, .allow_tab_completion = false});
                if (exec_result == CmdExecResult::quit) {
                    fmt::print("EOF [assumed Yes]");
                    return CmdExecResult::quit;
                }

                if (input.empty()) return CmdExecResult::done;

                using dvlab::str::tolower_string, dvlab::str::is_prefix_of;

                return (is_prefix_of(tolower_string(input), "yes"))
                           ? CmdExecResult::quit
                           : CmdExecResult::done;  // not yet to quit
            }};
}

Command history_cmd(CommandLineInterface& cli) {
    return {"history",
            [](ArgumentParser& parser) {
                parser.description("print command history");
                parser.option_prefix("+-");
                parser.add_argument<size_t>("num")
                    .default_value(SIZE_MAX)
                    .help("if specified, print the `num` latest command history");

                parser.add_argument<bool>("-c", "--clear")
                    .action(store_true)
                    .help("clear the command history");

                parser.add_argument<std::string>("-o", "--output")
                    .metavar("file")
                    .constraint(path_writable)
                    .help("output the command history to a file");

                parser.add_argument<bool>("--no-append-quit")
                    .action(store_true)
                    .help("don't append the quit command to the output. This argument has no effect if --output is not specified");

                auto success_mutex = parser.add_mutually_exclusive_group();
                success_mutex.add_argument<bool>("+s", "--include-success")
                    .action(store_true)
                    .help("include successful commands in the history");
                success_mutex.add_argument<bool>("-s", "--exclude-success")
                    .action(store_true)
                    .help("exclude successful commands in the history");

                auto error_mutex = parser.add_mutually_exclusive_group();
                error_mutex.add_argument<bool>("+e", "--include-errors")
                    .action(store_true)
                    .help("include commands returning errors in the history");

                error_mutex.add_argument<bool>("-e", "--exclude-errors")
                    .action(store_true)
                    .help("exclude commands returning errors in the history");

                auto unknown_mutex = parser.add_mutually_exclusive_group();

                unknown_mutex.add_argument<bool>("+u", "--include-unknowns")
                    .action(store_true)
                    .help("include unknown commands in the history");

                unknown_mutex.add_argument<bool>("-u", "--exclude-unknowns")
                    .action(store_true)
                    .help("exclude unknown commands in the history");

                auto interrupt_mutex = parser.add_mutually_exclusive_group();

                interrupt_mutex.add_argument<bool>("+i", "--include-interrupts")
                    .action(store_true)
                    .help("include interrupted commands in the history");

                interrupt_mutex.add_argument<bool>("-i", "--exclude-interrupts")
                    .action(store_true)
                    .help("exclude interrupted commands in the history");
            },
            [&cli](ArgumentParser const& parser) {
                auto num            = parser.get<size_t>("num");
                auto no_append_quit = parser.get<bool>("--no-append-quit");

                bool include_successes  = true;
                bool include_errors     = !parser.parsed("--output");
                bool include_unknowns   = !parser.parsed("--output");
                bool include_interrupts = !parser.parsed("--output");

                if (parser.parsed("+s")) include_successes = true;
                if (parser.parsed("-s")) include_successes = false;
                if (parser.parsed("+e")) include_errors = true;
                if (parser.parsed("-e")) include_errors = false;
                if (parser.parsed("+u")) include_unknowns = true;
                if (parser.parsed("-u")) include_unknowns = false;
                if (parser.parsed("+i")) include_interrupts = true;
                if (parser.parsed("-i")) include_interrupts = false;

                auto filter = CommandLineInterface::HistoryFilter{include_successes, include_errors, include_unknowns, include_interrupts};

                if (parser.parsed("--clear")) {
                    cli.clear_history();
                    return CmdExecResult::done;
                }
                if (parser.parsed("--output")) {
                    cli.write_history(parser.get<std::string>("--output"), num, !no_append_quit, filter);
                    return CmdExecResult::done;
                }

                cli.print_history(num, filter);

                return CmdExecResult::done;
            }};
}

Command source_cmd(CommandLineInterface& cli) {
    return {"source",
            [](ArgumentParser& parser) {
                parser.description("execute the commands in the dofile");

                parser.add_argument<bool>("-q", "--quiet")
                    .action(store_true)
                    .help("suppress the echoing of commands");
                parser.add_argument<std::string>("file")
                    .constraint(path_readable)
                    .help("path to a dofile, i.e., a list of qsyn commands");

                parser.add_argument<std::string>("arguments")
                    .nargs(NArgsOption::zero_or_more)
                    .help("arguments to the dofile");
            },
            [&cli](ArgumentParser const& parser) {
                auto arguments = parser.get<std::vector<std::string>>("arguments");
                return cli.source_dofile(parser.get<std::string>("file"), arguments, !parser.get<bool>("--quiet"));
            }};
}

Command usage_cmd(CommandLineInterface& cli) {
    return {"usage",
            [](ArgumentParser& parser) {
                parser.description("report the runtime and/or memory usage");

                auto mutex = parser.add_mutually_exclusive_group();

                mutex.add_argument<bool>("-t", "--time")
                    .action(store_true)
                    .help("print only time usage");
                mutex.add_argument<bool>("-m", "--memory")
                    .action(store_true)
                    .help("print only memory usage");
                mutex.add_argument<bool>("-r", "--reset")
                    .action(store_true)
                    .help("reset the period time usage counter");
            },
            [&](ArgumentParser const& parser) {
                auto reset = parser.get<bool>("--reset");

                if (reset) {
                    cli.usage().reset_period();
                    return CmdExecResult::done;
                }
                auto rep_time = parser.get<bool>("--time");
                auto rep_mem  = parser.get<bool>("--memory");
                if (!rep_time && !rep_mem) {
                    rep_time = true;
                    rep_mem  = true;
                }

                cli.usage().report(rep_time, rep_mem);

                return CmdExecResult::done;
            }};
}

Command logger_cmd() {
    static auto const log_levels = std::vector<std::string>{"off", "critical", "error", "warning", "info", "debug", "trace"};
    return Command{
        "logger",
        [](ArgumentParser& parser) {
            parser.description("display and set the logger's status");

            auto mutex = parser.add_mutually_exclusive_group();

            mutex.add_argument<bool>("-t", "--test")
                .action(store_true)
                .help("test current logger setting");

            mutex.add_argument<std::string>("level")
                .nargs(NArgsOption::optional)
                .constraint(choices_allow_prefix(log_levels))
                .help(fmt::format("set log levels. Levels (ascending): {}", fmt::join(log_levels, ", ")));
        },
        [](ArgumentParser const& parser) {
            if (parser.parsed("level")) {
                auto level = spdlog::level::from_str(parser.get<std::string>("level"));
                spdlog::set_level(level);
                spdlog::info("Setting logger level to \"{}\"", spdlog::level::to_string_view(level));
                return CmdExecResult::done;
            }
            if (parser.parsed("--test")) {
                spdlog::log(spdlog::level::level_enum::off, "Regular printing (level `{}`)", log_levels[0]);
                spdlog::critical("A log message with level `{}`", log_levels[1]);
                spdlog::error("A log message with level `{}`", log_levels[2]);
                spdlog::warn("A log message with level `{}`", log_levels[3]);
                spdlog::info("A log message with level `{}`", log_levels[4]);
                spdlog::debug("A log message with level `{}`", log_levels[5]);
                spdlog::trace("A log message with level `{}`", log_levels[6]);
                return CmdExecResult::done;
            }

            fmt::println("Logger Level: {}", spdlog::level::to_string_view(spdlog::get_level()));

            return CmdExecResult::done;
        }};
}

Command clear_cmd() {
    return {"clear",
            [](ArgumentParser& parser) {
                parser.description("clear the terminal");
            },

            [](ArgumentParser const& /*parser*/) {
                utils::clear_terminal();
                return CmdExecResult::done;
            }};
};

bool add_cli_common_cmds(dvlab::CommandLineInterface& cli) {
    if (!(cli.add_command(alias_cmd(cli)) &&
          cli.add_alias("unalias", "alias -d") &&
          cli.add_command(set_variable_cmd(cli)) &&
          cli.add_alias("unset", "set -d") &&
          cli.add_command(echo_cmd()) &&
          cli.add_command(quit_cmd(cli)) &&
          cli.add_command(history_cmd(cli)) &&
          cli.add_command(help_cmd(cli)) &&
          cli.add_command(source_cmd(cli)) &&
          cli.add_command(usage_cmd(cli)) &&
          cli.add_command(clear_cmd()) &&
          cli.add_command(logger_cmd()))) {
        spdlog::critical("Registering \"cli\" commands fails... exiting");
        return false;
    }
    return true;
}

}  // namespace dvlab
