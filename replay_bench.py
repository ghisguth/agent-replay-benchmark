#!/usr/bin/env python3
"""Agent-replay benchmark - CLI entry point.

Endpoints are OpenAI-compatible base URLs given with --endpoint; test cases live under bench/cases/.
Run `python replay_bench.py --help` for usage, or `--list` to preview the
endpoints/models the current selection resolves to.
"""

import argparse
import os
import sys
import textwrap

from bench import auth, device, paths, report
from bench.cases import replay as replay_cases
from bench.cli_args import parse_max_context, parse_run_note, parse_server_version, parse_turns_range
from bench.colors import Clr
from bench.discovery import describe_targets, discover_and_report
from bench.endpoints import DEFAULT_ENDPOINT, parse_endpoint, resolve_endpoints
from bench.outputs import output
from bench.presets import (
    DEFAULT_PRESET,
    PRESETS,  # noqa: F401
    SUITES,
    UNSET,
    parse_preset,
    preset_line,
    preset_size,
    presets_help,
    resolve_selection,
)
from bench.progress import ProgressAnimation
from bench.resume import apply_resume, resolve_state
from bench.runner import run_benchmark
from bench.tls import configure as configure_tls
from bench.utils import UserError, version_string


def parse_suite(value):
    v = str(value).strip().lower()
    if v not in SUITES:
        raise argparse.ArgumentTypeError("invalid suite %r (choose from %s)" % (value, ", ".join(SUITES)))
    return v


REPO = paths.ROOT
STATES_DIR = paths.default_states_dir()  # gitignored; every live run saves its JSONL state here


def dry_run(replay, suite="coding", mdcat=False):
    """Print the selected replay requests (no discovery, no network). Returns the exit code."""
    try:
        fixture = replay_cases.load_fixture(replay["fixture"])
    except (OSError, ValueError) as e:
        raise UserError("cannot read fixture %s: %s" % (replay["fixture"], getattr(e, "strerror", None) or e))
    selected = replay_cases.select_entries(
        fixture, max_context=replay["max_context"], no_subagents=replay["no_subagents"],
        turns=replay["turns"])
    if not selected:
        raise UserError("the selection is empty (check --max-context, --turns and --no-subagents)")
    output().dry_run(selected, "%s selection: %s" % (suite, replay_cases.summary_line(selected)), mdcat)
    return 0


def _prog_name():
    """The command as the user typed it; the launcher and `python3 replay_bench.py` show as replay-bench."""
    name = os.path.basename(sys.argv[0] or "")
    return "replay-bench" if not name or name.endswith(".py") else name


def main(argv=None):
    """Run the CLI. A UserError (a missing file, a bad option combination, no reachable server) is one line on stderr
    and exit code 2; any other unexpected exception is one line and exit code 1. REPLAY_BENCH_DEBUG=1 shows the
    traceback instead."""
    try:
        _main(argv)
    except UserError as e:
        if os.environ.get("REPLAY_BENCH_DEBUG"):
            raise
        report.error(str(e))
        sys.exit(2)
    except KeyboardInterrupt:
        sys.exit(130)
    except BrokenPipeError:
        # The reader of our output (`| head`) went away: leave quietly. Point stdout at devnull so the flush at
        # interpreter exit does not fail again.
        try:
            os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        except (OSError, ValueError):
            pass
        sys.exit(0)
    except SystemExit:
        raise
    except Exception as e:
        if os.environ.get("REPLAY_BENCH_DEBUG"):
            raise
        report.error("%s: %s (REPLAY_BENCH_DEBUG=1 shows the traceback)" % (type(e).__name__, e))
        sys.exit(1)


class BenchmarkHelpFormatter(argparse.RawDescriptionHelpFormatter):
    """Keeps explicit newlines and indentation in option help, and raw description/epilog."""

    def _split_lines(self, text, width):
        lines = []
        for line in text.splitlines():
            if not line.strip():
                lines.append("")
                continue
            indent = line[: len(line) - len(line.lstrip())]
            lines.extend(textwrap.wrap(line, width, subsequent_indent=indent, break_on_hyphens=False))
        return lines


class VersionAction(argparse.Action):
    """--version: the version is looked up (and git asked) only when the option is given."""

    def __init__(self, option_strings, dest="version", default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print("%s %s" % (parser.prog, version_string()))
        parser.exit()


class BenchmarkParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write("error: %s (see --help)\n" % message)
        sys.exit(2)


DESCRIPTION = """\
Replay a real coding-agent session turn by turn against OpenAI-compatible servers
(llama.cpp, Splash, Kapitsa, MTPLX, vLLM, LM Studio, ...). Every request of the
recorded session (main agent, subagents, compactions) is sent with the recorded
history, so the server always sees a realistic long context; each answer is graded
against what the agent did and what the reference models did at the same point;
speed (TTFT, prefill, decode), cache use and failures are measured per request.
Runs are saved as state files that can be shown again and compared with another run:
another model, quant, server or server version.
"""

EXAMPLES = """\
Examples:

  Quick sanity check of a server, four small generic cases:
    replay_bench.py --suite smoke --endpoint http://localhost:8080/v1

  The default (--preset large: main agent up to 128k) against llama.cpp, state saved to states/:
    replay_bench.py --endpoint http://localhost:8080/v1

  A router serving several models: pick by substring (case-insensitive), or take the first:
    replay_bench.py --endpoint http://localhost:8080/v1 --model Qwen3.8-27B --model Q4_K_M
    replay_bench.py --endpoint http://localhost:8080/v1 --first-model

  Check what a server reports (models, quant, context, version, device), no chat request:
    replay_bench.py --discover --endpoint http://localhost:8080/v1 --model Qwen3.8-27B

  What a preset or a custom selection would send, without a server:
    replay_bench.py --preset medium --dry-run
    replay_bench.py --suite coding --max-context 128k --no-subagents --turns 1-40 --dry-run

  Save to a named file, continue it after Ctrl+C or a crash (graded cases not sent again):
    replay_bench.py --endpoint http://localhost:8080/v1 --state-file states/run.jsonl
    replay_bench.py --resume states/run.jsonl

  Show a saved run again, as a markdown report written to a file:
    replay_bench.py --state run.jsonl
    replay_bench.py --state run.jsonl --report-format markdown > report.md

  Compare two runs case by case (another model, quant, server or server version):
    replay_bench.py --state new.jsonl --compare-state base.jsonl
    replay_bench.py --endpoint http://localhost:8080/v1 --compare-state base.jsonl

  Label what the server cannot tell (speculative decoding, build) so runs stay apart:
    replay_bench.py --endpoint http://localhost:8080/v1 --run-note MTP --server-version 1.6.0-m1\
"""


def build_epilog():
    presets_section = "Presets (--preset NAME; default: %s):\n%s" % (DEFAULT_PRESET, presets_help())
    notes_section = (
        "Notes:\n"
        "  Endpoints are http:// or https:// base URLs ending in /v1 (/v1 is appended when missing).\n"
        "  Environment variables:\n"
        "    OPENAI_API_KEY          Bearer token for endpoints that require authentication\n"
        "    NO_COLOR                Disable colors (also when stdout is not a terminal)\n"
        "    REPLAY_BENCH_DEBUG=1    Show tracebacks on error, and log every request URL\n"
        "    REPLAY_BENCH_STALL_SECONDS  Seconds without data from a streaming server before the request is\n"
        "                            given up as stalled (default 300)\n"
        "    REPLAY_BENCH_SERVER_WAIT  Seconds to wait for a vanished server per retry (default 900)\n"
        "    REPLAY_BENCH_NERD_FONT  1 to enable Nerd Font icons (same as --icons; 0 forces them off)\n"
        "  Exit codes: 0 done, 2 usage or input error, 1 unexpected error, or --list found no model.\n"
        "  Docs: README.md and docs/capture-traces.md."
    )
    return "\n\n".join([presets_section, notes_section, EXAMPLES])


def build_parser():
    parser = BenchmarkParser(
        prog=_prog_name(),
        usage="%(prog)s [options]",
        description=DESCRIPTION,
        formatter_class=BenchmarkHelpFormatter,
        add_help=False,
        epilog=build_epilog(),
    )
    g = parser.add_argument_group("Server & model")
    g.add_argument("--endpoint", action="append", type=parse_endpoint, metavar="URL", dest="endpoint",
                   help="OpenAI-compatible base URL (repeatable)\ndefault: %s" % DEFAULT_ENDPOINT)
    g.add_argument("--api-key", metavar="KEY", dest="api_key", default=None,
                   help="Bearer token for the server\ndefault: $OPENAI_API_KEY, else none")
    g.add_argument("--model", action="append", metavar="SUBSTRING",
                   help="Only models whose id contains this (case-insensitive, repeatable);\n"
                        "exact:NAME matches the whole id instead (so exact:Foo skips Foo:xxx)\ndefault: every model served")
    g.add_argument("--first-model", action="store_true", dest="first_model",
                   help="Only the first model each endpoint lists (after --model)")
    g.add_argument("--ca-file", metavar="FILE", dest="ca_file",
                   help="Extra CA bundle to trust for https:// endpoints\ndefault: system trust store only")
    g.add_argument("--insecure", action="store_true",
                   help="Do not verify the certificate of https:// endpoints")
    g.add_argument("--device", metavar="TEXT",
                   help="The server's hardware, e.g. \"MacBook Pro M2 Max 64GB\"\ndefault: detected when the server runs on this machine")
    g.add_argument("--server-version", metavar="VERSION", dest="server_version", type=parse_server_version,
                   help="Override the detected server version (e.g. 1.6.0-m1)\ndefault: detected from the server")
    g.add_argument("--run-note", metavar="TEXT", dest="run_note", type=parse_run_note,
                   help="A short note on the run, e.g. MTP or DFlash2\ndefault: none")

    g = parser.add_argument_group("Suite & selection")
    g.add_argument("--preset", default=None, metavar="NAME", type=parse_preset,
                   help="Named suite + selection (table under \"Presets\" below)\ndefault: %s\nAn explicit option that contradicts the preset is an error" % DEFAULT_PRESET)
    coding_n = preset_size("full")[0] or 470
    g.add_argument("--suite", default=None, metavar="NAME", type=parse_suite,
                   help="What to replay:\n  smoke   the first four sanity cases\n  sanity  ten generic cases: format, JSON, tools, math, recall (no replay)\n  coding  the recorded coding session (%d requests)\n  judge   one recorded match-judge case" % coding_n)
    g.add_argument("--fixture", metavar="FILE", default=None,
                   help="Replay this fixture instead of the suite's own\ndefault: the suite's fixture\nnot with: smoke and sanity suites")
    g.add_argument("--max-context", type=parse_max_context, default=UNSET, metavar="LIMIT",
                   dest="max_context",
                   help="Only turns that fit a server with this much context\ndefault: full (no limit; k=1024)\nchoices: 32k, 64k, 128k, 236k, full, or a token count\nnot with: smoke and sanity suites")
    g.add_argument("--no-subagents", action="store_true", dest="no_subagents",
                   help="Skip subagent sessions (replay only the main agent)\nnot with: smoke and sanity suites")
    g.add_argument("--turns", type=parse_turns_range, default=None, metavar="A-B",
                   help="Only requests A through B of the selection (inclusive, 1-based)\ndefault: all requests of the selection\nnot with: smoke and sanity suites")
    g.add_argument("--test", action="append", metavar="ID",
                   help="Run a specific smoke or sanity case by id (repeatable)\nnot with: replay suites")
    g.add_argument("--warmup-first-test", action="store_true",
                   help="Warm up with the first scored smoke or sanity test instead of a generic one\nnot with: replay suites")

    g = parser.add_argument_group("Sampling")
    g.add_argument("--temperature", type=float, default=None, metavar="T",
                   help="Sampling temperature to send\ndefault: 0.0 for smoke and sanity, none for replay suites\nnot with: --no-temperature")
    g.add_argument("--no-temperature", action="store_true",
                   help="Send no temperature field (server default applies)\ndefault for replay suites\nnot with: --temperature")
    g.add_argument("--reasoning-effort", metavar="LEVEL", dest="reasoning_effort", default=None,
                   help="reasoning_effort to send (e.g. low, medium, high)\ndefault: no field sent")

    g = parser.add_argument_group("State files & resume")
    g.add_argument("--state-file", metavar="FILE", dest="state_file",
                   help="Save results to FILE instead of the automatic path in states/\nnot with: --state, --no-save-state, --resume\n(see README, State files)")
    g.add_argument("--no-save-state", action="store_true", dest="no_save_state",
                   help="Do not save a state file (a live run saves under states/ otherwise)\nnot with: --state-file, --resume")
    g.add_argument("--state", metavar="FILE", dest="load_state", action="append",
                   help="Show one saved run again without sending anything (schema 1 JSONL); FILE is a path or a name (substring) in states/, fixtures/refs/ or fixtures/samples/\nnot with: --state-file, --no-save-state, --resume\n(see README, Comparing runs)")
    g.add_argument("--resume", metavar="FILE",
                   help="Continue an interrupted run from its state file\nnot with: --state, --state-file, --no-save-state, --compare-state, --dry-run, --list\n(see README, State files)")

    g = parser.add_argument_group("Comparison")
    g.add_argument("--compare-state", metavar="FILE", dest="compare_state",
                   help="Compare this run with a saved baseline run FILE (a path or a name in states/, fixtures/refs/ or fixtures/samples/)\nnot with: --resume, --report-format chat, chat-compact (tabular)\n(see README, Comparing runs)")

    g = parser.add_argument_group("Output")
    g.add_argument("--report-format", choices=report.FORMATS, default="tui", dest="report_format", metavar="F",
                   help="tui (default)  tables with colors\nmarkdown       everything as markdown (> out.md gives a clean file)\nchat           the replay as an agent transcript, then the tables\nchat-compact   compact transcript with one live line")
    g.add_argument("--no-mdcat", action="store_true", dest="no_mdcat",
                   help="Do not render markdown through mdcat/glow on a terminal\ndefault: mdcat when installed and stdout is a terminal")
    g.add_argument("--icons", action="store_true", dest="icons",
                   help="Use Nerd Font icons in chat formats on a color terminal (needs a Nerd Font)\ndefault: off, plain Unicode (or env REPLAY_BENCH_NERD_FONT=1)")
    g.add_argument("--no-nerd-font", action="store_true", dest="no_nerd_font",
                   help="Force plain Unicode, even if REPLAY_BENCH_NERD_FONT=1\ndefault: plain Unicode")
    g.add_argument("--no-prompt-progress", action="store_true", dest="no_prompt_progress",
                   help="Do not ask servers that report prefill progress for it\ndefault: on (Splash, llama.cpp, Kapitsa, MTPLX)")

    g = parser.add_argument_group("Diagnostics")
    g.add_argument("-h", "--help", action="help", help="Show this help and exit")
    g.add_argument("--version", action=VersionAction,
                   help="Show the version (and the git commit in a checkout) and exit")
    g.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print selected requests and exit (no network, no server needed)")
    g.add_argument("--list", "--discover", action="store_true", dest="list",
                   help="Probe endpoints, show what the servers report for the selected models, then exit\nshows: quant, context, family, runner per model; version, build, device, prefill progress per server\nsends no chat request; exit code 1 when no model matched, else 0")

    return parser


def _main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    auth.set_api_key(args.api_key)
    resume = None
    if args.resume:
        args, resume = apply_resume(parser, args, argv)
    report.set_nerd_font(True if args.icons else (False if args.no_nerd_font else None))

    # Validate incompatible flag combos
    if args.compare_state and args.report_format in report.CHAT_FORMATS:
        parser.error("--compare-state is not supported with --report-format %s (use tui or markdown)" % args.report_format)
    if args.load_state and len(args.load_state) > 1:
        parser.error("--state takes one file (it shows one saved run); to compare two runs use --state A --compare-state B")
    if args.load_state:
        if (args.preset is not None or args.turns is not None
                or args.max_context is not UNSET or args.no_subagents
                or args.suite is not None):
            parser.error("selection options don't apply to --state; the saved run is shown as recorded")
        args.load_state = resolve_state(args.load_state[0], states_dir=STATES_DIR)
    if args.compare_state:
        args.compare_state = resolve_state(args.compare_state, states_dir=STATES_DIR)
    if args.load_state and args.state_file:
        parser.error("--state and --state-file are mutually exclusive.")
    if args.state_file and args.no_save_state:
        parser.error("--state-file and --no-save-state are mutually exclusive.")

    # No suite, preset or fixture: the default preset (the smoke suite for --test, --state, --list, which need no replay).
    preset, suite_key, max_context, no_subagents, turns = resolve_selection(
        args, default=not (args.load_state or args.list))
    suite = SUITES[suite_key]  # None: smoke

    selection_given = [flag for flag, given in (
        ("--max-context", args.max_context is not UNSET), ("--no-subagents", args.no_subagents),
        ("--turns", args.turns is not None), ("--fixture", args.fixture is not None),
        ("--dry-run", args.dry_run)) if given]
    if not suite and selection_given:
        parser.error("%s can only be used with a replay suite (--suite coding or judge, or a replay preset), "
                     "not with the smoke or sanity suites / --test." % ", ".join(selection_given))
    if suite and args.test:
        parser.error("--test selects smoke or sanity cases and cannot be combined with --suite %s (use --turns)." % suite_key)
    args.max_context, args.no_subagents, args.turns = max_context, no_subagents, turns

    replay = None
    if suite:
        replay = {"fixture": args.fixture or replay_cases.default_fixture(suite),
                  "max_context": args.max_context, "no_subagents": args.no_subagents,
                  "turns": args.turns}
        if args.dry_run:
            report.configure(args.report_format)
            if preset:
                print(preset_line(preset), file=sys.stderr)
            sys.exit(dry_run(replay, suite, mdcat=not args.no_mdcat))
    if preset and not args.load_state and not args.list:
        print(preset_line(preset), file=sys.stderr)

    configure_tls(ca_file=args.ca_file, insecure=args.insecure)
    device.configure(args.device)
    try:
        endpoints = resolve_endpoints(args.endpoint)
    except argparse.ArgumentTypeError as e:
        parser.error(str(e))

    report.configure(args.report_format)
    if args.list:
        targets = discover_and_report(endpoints, args.model, spinner=ProgressAnimation(), first_model=args.first_model)
        if not targets:
            print(f"{Clr.BYEL}No reachable models matched the current selection.{Clr.RST}")
            sys.exit(1)
        describe_targets(targets)
        sys.exit(0)

    # smoke is the default suite; sanity is the other one without a replay
    mode = suite or suite_key
    if args.no_temperature:
        temperature = None
    elif args.temperature is not None:
        temperature = args.temperature
    else:
        temperature = None if replay else 0.0

    run_benchmark(
        mode,
        target_test_ids=args.test,
        endpoints=endpoints,
        model_filters=args.model,
        first_model=args.first_model,
        state_file=args.state_file,
        save_state=not args.no_save_state,
        resume=resume,
        states_dir=STATES_DIR,
        cmdline=list(sys.argv[1:] if argv is None else argv),
        compare_state_path=args.compare_state,
        load_state_path=args.load_state,
        mdcat=not args.no_mdcat,
        warmup_first_test=args.warmup_first_test,
        reasoning_effort=args.reasoning_effort,
        temperature=temperature,
        replay=replay,
        preset=preset,
        report_format=args.report_format,
        prompt_progress=not args.no_prompt_progress,
        run_note=args.run_note,
        server_version=args.server_version,
    )


if __name__ == "__main__":
    main()
