"""Import shim: makes `old_code/` importable under the package name it expects.

old_code's modules import each other as `features._common`,
`features.inference_lib.door_tracker`, `features.door.processor` and so on. Rather
than editing 15 files -- which would fork the behavioural source of truth and
defeat the point of porting -- this package extends its own search path to
`old_code/`, so every one of those imports resolves to the ORIGINAL file, byte for
byte.

Consequences worth being explicit about:

  * `old_code/` stays the single copy of the mature algorithms. There is no second,
    drifting implementation to keep in sync.
  * Any behavioural difference from the old system therefore comes from the
    compatibility layer in `core/` (documented there) or from the frame cache that
    feeds the processors -- never from a silent edit to a tracker.
  * The old modules keep their own imports, so `scipy` (Hungarian assignment) and
    `easyocr` (OCR) remain their real dependencies; a missing one degrades exactly
    as old_code degrades it.
"""

import os as _os

_OLD_CODE = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "old_code")

if _os.path.isdir(_OLD_CODE):
    # Prepend so old_code's modules win, and keep this package's own directory on
    # the path in case project-local overrides are ever added alongside.
    __path__ = [_OLD_CODE] + list(__path__)
