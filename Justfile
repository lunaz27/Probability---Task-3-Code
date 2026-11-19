# NOTE: Documentation: https://just.systems

set quiet

default:
    just --list

run:
    python -m src.main

lint args="" +files=".":
    echo "  LINT    tests/*.py"
    ruff check {{ files }} {{ args }}

format +files="": lint
    echo "  FORMAT  *.py"
    ruff format {{ files }}

test: lint format
    echo "  PYTEST  tests/"
    pytest
    echo "  PASSED  pytest"

entr:
    #!/usr/bin/env bash
    fd . src/ tests/ | entr -c bash -c "just run"
