"""Tests for the unified Streamlit entrypoint helpers."""

from __future__ import annotations

from query_app import build_workspace_options, format_workspace_label


def test_build_workspace_options_returns_expected_order() -> None:
    """The unified site should expose dashboard first, then graph mode."""

    assert build_workspace_options() == ["dashboard", "graph"]


def test_format_workspace_label_maps_internal_ids_to_titles() -> None:
    """Workspace labels should stay readable in the sidebar radio control."""

    assert format_workspace_label("dashboard") == "Analysis Dashboard"
    assert format_workspace_label("graph") == "Interactive Graph"
