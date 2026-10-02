"""The tools that read scores and never compute one.

Nothing under this directory enters `eval_code_sha`: a fix to one of these
tools, or a new one, must not make old and new scores incomparable, and no
score depends on them. `paired_stats`, `compare_eval` and
`condition_inventory` are ported here. A module that decides a score does
not belong here; it goes beside `code_sha.py`, on its list.
"""
