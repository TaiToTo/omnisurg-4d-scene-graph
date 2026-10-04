"""The evaluator, and under `tools/` the tools that read its scores.

`docs/evaluation.md` is the specification. Every score carries
`eval_code_sha`, a hash of every module here and of the class tables. The
tools that only read scores (`tools/paired_stats`, `tools/compare_eval`, ...)
are not hashed: fixing one of them, or adding one, must not make old and
new scores incomparable. The files under the freeze rule are exactly the
hashed ones.
"""
