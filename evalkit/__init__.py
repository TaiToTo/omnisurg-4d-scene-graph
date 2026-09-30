"""The evaluator, and the tools that read its scores.

`docs/evaluation.md` is the specification. Every score carries
`eval_code_sha`, a hash of the modules that compute scores and of the class
tables. The tools that only read scores (`paired_stats`, `compare_eval`, ...)
are not hashed: fixing one of them must not make old and new scores
incomparable. The files under the freeze rule are exactly the hashed ones.
"""
