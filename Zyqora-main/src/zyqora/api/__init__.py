"""Zyqora web API (FastAPI) on top of the existing Zyqora pipeline.

The API never runs its own inference code: it uses zyqora.profiles (model load
with contract/decoder checks), zyqora.pipeline (preprocess -> ExecuTorch ->
decoder -> EcoGuard), zyqora.sorter, zyqora.greenscore and zyqora.voicesort.
"""
