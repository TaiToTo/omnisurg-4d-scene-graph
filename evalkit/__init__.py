"""The evaluator, and the toolkit that reads its scores.

`docs/evaluation.md` is the specification. Every score the evaluator writes
carries `eval_code_sha`, a hash of this package's code and class tables, so
that two scores are compared only when they were made by the same evaluator.
"""
