"""Module entrypoint enabling ``python -m pymts`` executions."""

from __future__ import annotations

from cli.app import app


def main() -> None:
    """Invoke the Typer application."""

    app()


if __name__ == "__main__":
    main()
