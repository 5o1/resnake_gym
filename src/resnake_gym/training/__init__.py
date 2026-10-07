"""Reusable training orchestration helpers.

Command-line programs in ``scripts/`` should parse arguments and invoke these
helpers; persistent experiment state belongs in the importable package so it
can be tested without loading a script by file path.
"""
