"""What the pipeline and the evaluation toolkit share.

Dataset access, geometry, preprocessing and the viewer's data model live
here, so that the pipeline that produces the scene graphs and the toolkit that
measures them read the same data through the same code. A rule that exists
once cannot be fixed on one side and not the other.
"""
