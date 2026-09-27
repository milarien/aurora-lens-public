from __future__ import annotations

import importlib.util
from pathlib import Path

from setuptools import setup
from setuptools.command.sdist import sdist as _sdist

_ROOT = Path(__file__).resolve().parent
_GUARD_PATH = _ROOT / "release_guard.py"
_SPEC = importlib.util.spec_from_file_location("release_guard", _GUARD_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover
    raise RuntimeError("Could not load release_guard.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
assert_clean_release_tree = _MODULE.assert_clean_release_tree

# Load the build backend so BUILD_INFO.json is written for direct setup.py invocations too.
_BACKEND_PATH = _ROOT / "aurora_lens_build_backend.py"
_BACKEND_SPEC = importlib.util.spec_from_file_location("aurora_lens_build_backend", _BACKEND_PATH)
if _BACKEND_SPEC is not None and _BACKEND_SPEC.loader is not None:
    _BACKEND_MODULE = importlib.util.module_from_spec(_BACKEND_SPEC)
    try:
        _BACKEND_SPEC.loader.exec_module(_BACKEND_MODULE)
        _write_build_info = getattr(_BACKEND_MODULE, "_write_build_info", None)
    except Exception:
        _write_build_info = None
else:
    _write_build_info = None

try:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel
except Exception:  # pragma: no cover - wheel not installed
    _bdist_wheel = None


class GuardedSdist(_sdist):
    def run(self) -> None:
        if _write_build_info is not None:
            _write_build_info()
        assert_clean_release_tree(_ROOT)
        super().run()


if _bdist_wheel is not None:
    class GuardedBdistWheel(_bdist_wheel):
        def run(self) -> None:
            if _write_build_info is not None:
                _write_build_info()
            assert_clean_release_tree(_ROOT)
            super().run()

    cmdclass = {"sdist": GuardedSdist, "bdist_wheel": GuardedBdistWheel}
else:
    cmdclass = {"sdist": GuardedSdist}


setup(cmdclass=cmdclass)
