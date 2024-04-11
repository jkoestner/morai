"""Provide cli for morai."""

import argparse
from argparse import ArgumentDefaultsHelpFormatter

from morai.dashboard import app


def _create_argparser():
    description = """Morai CLI."""
    _parser = argparse.ArgumentParser(
        description=description,
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    # creating subparsers for separate commands
    _subparsers = _parser.add_subparsers(dest="command", help="command to choose")

    # subparser: dashboard
    _dashboard_parser = _subparsers.add_parser("dashboard", help="dashboar command")

    return _parser


parser = _create_argparser()


def cli():
    """Command line interface."""
    args = parser.parse_args()

    if args.command == "dashboard":
        app.run_server(debug=False)


if __name__ == "__main__":
    cli()
