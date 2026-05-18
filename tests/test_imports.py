"""M0 smoke tests — verify package imports cleanly with no Qpress runtime deps."""
from __future__ import annotations


def test_package_import():
    import flake_core

    assert flake_core.__version__ == "0.1.0"


def test_compat_msg_shim():
    from flake_core._compat import msg

    # Should not raise
    msg.info("smoke")
    msg.debug("smoke")
    msg.warning("smoke")
    msg.error("smoke")


def test_compat_operation_context_shim():
    from flake_core._compat import OperationContext

    ctx = OperationContext(params={"key": "value"})
    assert ctx.params == {"key": "value"}
    assert ctx.inputs == {}
    assert ctx.state == {}


def test_compat_analysis_tree_noop():
    from flake_core._compat import AnalysisTree

    AnalysisTree.write_meta_json("any", "args")
    AnalysisTree.register()
    assert AnalysisTree.get_parent() is None
