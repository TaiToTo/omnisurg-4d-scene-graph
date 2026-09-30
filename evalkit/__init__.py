"""The evaluator, and the toolkit that reads its scores.

`docs/evaluation.md` is the specification. Every score the evaluator writes
carries `eval_code_sha`, a hash of the evaluator's modules and class tables.
Not of this whole package: the tools that read scores (`paired_stats`,
`compare_eval`, ...) live here too, and fixing one of them must not make
every score before it incomparable with every score after. The frozen files
are exactly the hashed ones.
"""
