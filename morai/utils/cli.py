"""Provide cli for morai."""

import argparse
from argparse import ArgumentDefaultsHelpFormatter

from morai.dashboard import app


def _create_argparser() -> argparse.ArgumentParser:
    description = """Morai CLI."""
    _parser = argparse.ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # creating subparsers for separate commands
    _subparsers = _parser.add_subparsers(dest="command", help="command to choose")

    # subparser: dashboard
    _dashboard_parser = _subparsers.add_parser("dashboard", help="dashboard command")

    return _parser


parser = _create_argparser()


def cli() -> None:
    """Command line interface."""
    args = parser.parse_args()

    if args.command == "dashboard":
        app.app.run_server(debug=True)


if __name__ == "__main__":
    cli()
