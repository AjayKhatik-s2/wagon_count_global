"""Compatibility package for the ported `old_code` feature processors.

old_code imports `core.constants`, `core.global_state_loader` and
`core.unified_wagon_state`. None shipped with it, so all three are reconstructed
here -- which is what allows the mature door / load / damage / OCR algorithms to
run VERBATIM instead of being rewritten.
"""
