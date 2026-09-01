"""Unified Streamlit entry point for the market-network web interface."""

from __future__ import annotations

from interactive_graph_app import run_app as run_graph_app
from streamlit_app import run_app as run_query_app

try:
    import streamlit as st
except ImportError:  # pragma: no cover - handled at runtime
    st = None


WORKSPACE_LABELS = {
    "dashboard": "Analysis Dashboard",
    "graph": "Interactive Graph",
}


def build_workspace_options() -> list[str]:
    """Return the ordered workspace modes for the unified web app."""

    return ["dashboard", "graph"]


def format_workspace_label(mode: str) -> str:
    """Map one internal workspace id to a user-facing label."""

    return WORKSPACE_LABELS[mode]


def render_workspace_switcher() -> str:
    """Render the sidebar workspace switcher and return the selected mode."""

    with st.sidebar:
        st.markdown("## Workspace")
        selected_mode = st.radio(
            "View",
            options=build_workspace_options(),
            format_func=format_workspace_label,
            help=(
                "Dashboard mode focuses on lookups, comparisons, paths, and LLM "
                "impact summaries. Graph mode focuses on draggable network structure."
            ),
        )
        st.caption(
            "One deployed site now includes both the query dashboard and the "
            "interactive graph explorer."
        )
    return selected_mode


def run_app() -> None:
    """Run the unified Streamlit app used for GitHub and Streamlit Cloud."""

    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )

    st.set_page_config(
        page_title="Stock and News Network Explorer",
        page_icon="",
        layout="wide",
    )
    selected_mode = render_workspace_switcher()

    if selected_mode == "dashboard":
        run_query_app(configure_page=False, embedded=True)
        return

    run_graph_app(configure_page=False, embedded=True)


if __name__ == "__main__":
    run_app()
