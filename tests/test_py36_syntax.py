import ast
import os
import sys

import pytest


def _iter_python_files():
    for root in ["pycodex", "responses_server", "tests"]:
        for dirpath, _dirnames, filenames in os.walk(root):
            for filename in filenames:
                if filename.endswith(".py"):
                    yield os.path.join(dirpath, filename)


def test_python_sources_parse_with_python36_grammar():
    if sys.version_info < (3, 8):
        pytest.skip("ast feature_version requires Python 3.8+")

    failures = []
    for path in sorted(_iter_python_files()):
        with open(path, "r", encoding="utf-8") as handle:
            source = handle.read()
        try:
            ast.parse(source, filename=path, feature_version=(3, 6))
        except SyntaxError as exc:
            failures.append("%s:%s: %s" % (path, exc.lineno, exc.msg))

    assert not failures, "Python 3.6-incompatible syntax found:\n%s" % "\n".join(failures)
