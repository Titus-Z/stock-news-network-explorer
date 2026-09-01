"""Streamlit app for draggable graph visualization of the market network."""

from __future__ import annotations

from html import escape
from typing import Any

import networkx as nx
import pandas as pd

from streamlit_app import (
    DEFAULT_WEB_TICKERS,
    load_analysis_bundle,
    normalize_top_k,
    parse_ticker_text,
)

try:
    import streamlit as st
    import streamlit.components.v1 as components
except ImportError:  # pragma: no cover - handled at runtime
    st = None
    components = None

try:
    from pyvis.network import Network
except ImportError:  # pragma: no cover - handled at runtime
    Network = None


STYLE_PROFILES = {
    "AWS + Bloom": {
        "hero_title": "Graph Explorer Workbench",
        "hero_body": (
            "A product-style graph workbench inspired by AWS Graph Explorer "
            "and Neo4j Bloom: clearer panes, stable semantic colors, and an "
            "investigation-first layout."
        ),
        "page_radial_left": "rgba(37, 99, 235, 0.12)",
        "page_radial_right": "rgba(15, 118, 110, 0.10)",
        "page_base_start": "#F7F9FC",
        "page_base_end": "#EEF2F7",
        "sidebar_start": "#111827",
        "sidebar_end": "#0F172A",
        "metric_bg": "rgba(255, 255, 255, 0.90)",
        "metric_border": "rgba(148, 163, 184, 0.18)",
        "hero_start": "rgba(15, 23, 42, 0.96)",
        "hero_end": "rgba(37, 99, 235, 0.94)",
        "card_bg": "rgba(255, 255, 255, 0.92)",
        "card_border": "rgba(148, 163, 184, 0.16)",
        "legend_bg": "rgba(255, 255, 255, 0.94)",
        "network_bg": "#F8FBFF",
        "network_font": "#111111",
        "canvas_caption": (
            "Drag nodes, zoom, and use the built-in graph menus. This style is "
            "meant to feel like a graph investigation product, not a loose demo."
        ),
        "profile_summary": (
            "Best when you want the app to read like a polished graph analytics "
            "tool with clean panes and a structured investigation flow."
        ),
        "node_colors": {
            "stock": "#2F6BFF",
            "sector": "#14906B",
            "topic": "#F08C2E",
            "unknown": "#7F8C8D",
        },
        "edge_colors": {
            "stock_stock": "#2563EB",
            "stock_sector": "#0F766E",
            "stock_topic": "#D97706",
            "sector_sector": "#9333EA",
            "unknown": "#B0B8C5",
        },
        "legend_accents": ["#2563EB", "#0F766E", "#D97706"],
    },
    "AWS + Sigma": {
        "hero_title": "Modern Network Canvas",
        "hero_body": (
            "A cleaner, more technical graph canvas inspired by Sigma-style "
            "WebGL exploration layered into a productized explorer shell."
        ),
        "page_radial_left": "rgba(14, 165, 233, 0.12)",
        "page_radial_right": "rgba(59, 130, 246, 0.10)",
        "page_base_start": "#F8FBFF",
        "page_base_end": "#EEF5FF",
        "sidebar_start": "#0B1120",
        "sidebar_end": "#111827",
        "metric_bg": "rgba(255, 255, 255, 0.95)",
        "metric_border": "rgba(96, 165, 250, 0.18)",
        "hero_start": "rgba(2, 6, 23, 0.98)",
        "hero_end": "rgba(37, 99, 235, 0.92)",
        "card_bg": "rgba(255, 255, 255, 0.96)",
        "card_border": "rgba(148, 163, 184, 0.14)",
        "legend_bg": "rgba(255, 255, 255, 0.98)",
        "network_bg": "#FFFFFF",
        "network_font": "#0F172A",
        "canvas_caption": (
            "This style favors a cleaner canvas, stronger contrast, and a "
            "lighter frame around the network for a more modern graph-browser feel."
        ),
        "profile_summary": (
            "Best when you want the graph itself to dominate, with less visual "
            "weight in the surrounding panels."
        ),
        "node_colors": {
            "stock": "#2563EB",
            "sector": "#0EA5E9",
            "topic": "#7C3AED",
            "unknown": "#94A3B8",
        },
        "edge_colors": {
            "stock_stock": "#1D4ED8",
            "stock_sector": "#0F766E",
            "stock_topic": "#8B5CF6",
            "sector_sector": "#0284C7",
            "unknown": "#CBD5E1",
        },
        "legend_accents": ["#1D4ED8", "#0EA5E9", "#7C3AED"],
    },
    "Kumu + Bloom": {
        "hero_title": "Systems Map Explorer",
        "hero_body": (
            "A softer systems-map treatment inspired by Kumu, while keeping "
            "Bloom-like semantic clarity for graph exploration."
        ),
        "page_radial_left": "rgba(16, 185, 129, 0.10)",
        "page_radial_right": "rgba(245, 158, 11, 0.08)",
        "page_base_start": "#FBFBF7",
        "page_base_end": "#F3F2EA",
        "sidebar_start": "#1F2937",
        "sidebar_end": "#111827",
        "metric_bg": "rgba(255, 255, 255, 0.92)",
        "metric_border": "rgba(163, 163, 163, 0.18)",
        "hero_start": "rgba(20, 83, 45, 0.94)",
        "hero_end": "rgba(180, 83, 9, 0.88)",
        "card_bg": "rgba(255, 255, 250, 0.92)",
        "card_border": "rgba(161, 161, 170, 0.16)",
        "legend_bg": "rgba(255, 255, 252, 0.96)",
        "network_bg": "#FCFCF8",
        "network_font": "#1F2937",
        "canvas_caption": (
            "This style softens the product shell and makes the graph read more "
            "like a relationship map or systems map."
        ),
        "profile_summary": (
            "Best when you want the project to feel more like a network map of "
            "market relationships than a technical graph IDE."
        ),
        "node_colors": {
            "stock": "#4F46E5",
            "sector": "#15803D",
            "topic": "#C2410C",
            "unknown": "#71717A",
        },
        "edge_colors": {
            "stock_stock": "#6366F1",
            "stock_sector": "#15803D",
            "stock_topic": "#C2410C",
            "sector_sector": "#A16207",
            "unknown": "#A1A1AA",
        },
        "legend_accents": ["#4F46E5", "#15803D", "#C2410C"],
    },
}
VISUAL_PRESETS = {
    "Market Structure": {
        "description": (
            "Full multi-layer graph view. Use this to see stocks, sectors, "
            "topics, and the bridge structure across all edge types."
        ),
        "node_types": {"stock", "sector", "topic"},
        "edge_types": {"stock_stock", "stock_sector", "stock_topic", "sector_sector"},
    },
    "Sector Bridges": {
        "description": (
            "Strips the graph down to sector relationships and the stocks that "
            "connect those sectors through strong correlation edges."
        ),
        "node_types": {"stock", "sector"},
        "edge_types": {"stock_stock", "stock_sector", "sector_sector"},
    },
    "Topic Exposure": {
        "description": (
            "Emphasizes how companies and sectors are exposed to news topics, "
            "while keeping sector membership visible for context."
        ),
        "node_types": {"stock", "sector", "topic"},
        "edge_types": {"stock_topic", "stock_sector"},
    },
    "Topic Map": {
        "description": (
            "A cleaner topic view for inspecting which stocks cluster around the "
            "same news themes without sector-to-sector structure."
        ),
        "node_types": {"stock", "topic"},
        "edge_types": {"stock_topic"},
    },
}

DEFAULT_STYLE_PROFILE = "AWS + Bloom"
DEFAULT_GRAPH_TICKERS = [
    "AAPL",
    "MSFT",
    "NVDA",
]


def build_style_profile_options() -> list[str]:
    """Return the supported graph visual styles in display order."""

    return list(STYLE_PROFILES)


def get_style_profile(profile_name: str | None) -> dict[str, Any]:
    """Resolve one style profile, falling back to the default option."""

    if profile_name and profile_name in STYLE_PROFILES:
        return STYLE_PROFILES[profile_name]
    return STYLE_PROFILES[DEFAULT_STYLE_PROFILE]


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert one hex color into an rgba() string for highlight states."""

    color = hex_color.lstrip("#")
    if len(color) != 6:
        return hex_color
    red = int(color[0:2], 16)
    green = int(color[2:4], 16)
    blue = int(color[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def node_color(
    node_type: str,
    style_profile: dict[str, Any] | None = None,
) -> str:
    """Return a stable color for one node type."""

    resolved_profile = style_profile or get_style_profile(None)
    node_colors = resolved_profile["node_colors"]
    return node_colors.get(node_type, node_colors["unknown"])


def node_size(node_data: dict[str, Any], degree: int) -> int:
    """Compute a readable node size from type and light metadata."""

    node_type = node_data.get("node_type")
    if node_type == "sector":
        return 26 + min(degree * 2, 12)
    if node_type == "stock":
        return 18 + min(degree * 2, 14)
    if node_type == "topic":
        article_count = int(node_data.get("article_count") or 0)
        return 12 + min(article_count // 10, 16)
    return 16


def edge_width(edge_data: dict[str, Any]) -> float:
    """Map edge metadata to a visible line width."""

    edge_type = edge_data.get("edge_type")
    if edge_type == "stock_stock":
        correlation = float(edge_data.get("correlation") or 0.0)
        return max(1.0, round(correlation * 6, 2))
    if edge_type == "stock_topic":
        article_count = int(edge_data.get("article_count") or 0)
        return max(1.0, min(6.0, 1.0 + article_count / 20))
    if edge_type == "sector_sector":
        average_correlation = float(edge_data.get("average_correlation") or 0.0)
        return max(1.0, round(average_correlation * 6, 2))
    return 1.5


def edge_color(
    edge_data: dict[str, Any],
    style_profile: dict[str, Any] | None = None,
) -> str:
    """Map edge type to a stable display color."""

    resolved_profile = style_profile or get_style_profile(None)
    edge_colors = resolved_profile["edge_colors"]
    edge_type = str(edge_data.get("edge_type", "unknown"))
    return edge_colors.get(edge_type, edge_colors["unknown"])


def build_node_title(node_id: str, node_data: dict[str, Any], degree: int) -> str:
    """Build tooltip text for one node."""

    node_type = node_data.get("node_type", "unknown")
    lines = [f"id: {node_id}", f"type: {node_type}", f"label: {node_data.get('label')}"]

    if node_type == "stock":
        lines.extend(
            [
                f"ticker: {node_data.get('ticker')}",
                f"company: {node_data.get('company_name') or 'Unknown'}",
                f"sector: {node_data.get('sector') or 'Unknown'}",
                f"industry: {node_data.get('industry') or 'Unknown'}",
            ]
        )
    elif node_type == "sector":
        lines.extend(
            [
                f"sector: {node_data.get('sector')}",
                f"stock_count: {node_data.get('stock_count')}",
            ]
        )
    elif node_type == "topic":
        lines.extend(
            [
                f"topic: {node_data.get('topic')}",
                f"article_count: {node_data.get('article_count')}",
            ]
        )

    lines.append(f"degree: {degree}")
    return "<br>".join(escape(str(line)) for line in lines)


def build_edge_title(edge_data: dict[str, Any]) -> str:
    """Build tooltip text for one edge."""

    edge_type = edge_data.get("edge_type", "unknown")
    lines = [f"edge_type: {edge_type}"]

    if edge_type == "stock_stock":
        lines.extend(
            [
                f"correlation: {edge_data.get('correlation')}",
                f"overlap_days: {edge_data.get('overlap_days')}",
            ]
        )
    elif edge_type == "stock_topic":
        lines.extend(
            [
                f"weight: {edge_data.get('weight')}",
                f"article_count: {edge_data.get('article_count')}",
                f"avg_ticker_sentiment: {edge_data.get('avg_ticker_sentiment')}",
            ]
        )
    elif edge_type == "sector_sector":
        lines.extend(
            [
                f"average_correlation: {edge_data.get('average_correlation')}",
                f"stock_edge_count: {edge_data.get('stock_edge_count')}",
            ]
        )
    else:
        lines.append(f"weight: {edge_data.get('weight')}")

    return "<br>".join(escape(str(line)) for line in lines)


def filter_graph(
    graph: nx.Graph,
    included_node_types: set[str],
    included_edge_types: set[str],
    remove_isolates: bool,
) -> nx.Graph:
    """Filter the graph by node and edge type selections."""

    filtered_graph = graph.copy()

    # Remove node types first so subsequent edge filtering only works with the
    # surviving part of the graph.
    node_ids_to_remove = [
        node_id
        for node_id, node_data in filtered_graph.nodes(data=True)
        if node_data.get("node_type") not in included_node_types
    ]
    filtered_graph.remove_nodes_from(node_ids_to_remove)

    edge_ids_to_remove = [
        (source, target)
        for source, target, edge_data in filtered_graph.edges(data=True)
        if edge_data.get("edge_type") not in included_edge_types
    ]
    filtered_graph.remove_edges_from(edge_ids_to_remove)

    if remove_isolates:
        isolate_nodes = list(nx.isolates(filtered_graph))
        filtered_graph.remove_nodes_from(isolate_nodes)

    return filtered_graph


def apply_focus_filter(
    graph: nx.Graph,
    focus_node_id: str | None,
    max_distance: int,
) -> nx.Graph:
    """Restrict the graph to one node neighborhood for focused inspection."""

    if not focus_node_id or focus_node_id not in graph:
        return graph

    reachable = nx.single_source_shortest_path_length(
        graph,
        source=focus_node_id,
        cutoff=max_distance,
    )
    return graph.subgraph(reachable.keys()).copy()


def build_focus_node_option_map(graph: nx.Graph) -> dict[str, str]:
    """Build readable focus-node labels for the explorer sidebar."""

    option_map: dict[str, str] = {}
    for node_id, node_data in sorted(graph.nodes(data=True)):
        label = str(node_data.get("label") or node_id)
        node_type = str(node_data.get("node_type", "unknown"))
        degree = graph.degree(node_id)
        option_map[f"{node_type} | {label} | degree {degree}"] = node_id
    return option_map


def build_shortest_path_result(
    graph: nx.Graph,
    source_node_id: str | None,
    target_node_id: str | None,
) -> dict[str, Any]:
    """Compute one shortest path summary for graph highlighting."""

    if not source_node_id or not target_node_id:
        return {"path_nodes": [], "path_edges": [], "status": "missing"}
    if source_node_id not in graph or target_node_id not in graph:
        return {"path_nodes": [], "path_edges": [], "status": "invalid"}
    if source_node_id == target_node_id:
        return {
            "path_nodes": [source_node_id],
            "path_edges": [],
            "status": "same-node",
        }

    try:
        path_nodes = nx.shortest_path(graph, source=source_node_id, target=target_node_id)
    except nx.NetworkXNoPath:
        return {"path_nodes": [], "path_edges": [], "status": "no-path"}

    path_edges = list(zip(path_nodes[:-1], path_nodes[1:]))
    return {
        "path_nodes": path_nodes,
        "path_edges": path_edges,
        "status": "ok",
    }


def build_path_summary_frame(graph: nx.Graph, path_result: dict[str, Any]) -> pd.DataFrame:
    """Convert one shortest path result into a readable summary table."""

    status = str(path_result.get("status", "missing"))
    path_nodes = list(path_result.get("path_nodes") or [])
    if status != "ok":
        return pd.DataFrame(
            [{"field": "status", "value": status.replace("-", " ")}]
        )

    labeled_path = " -> ".join(
        str(graph.nodes[node_id].get("label") or node_id)
        for node_id in path_nodes
    )
    return pd.DataFrame(
        [
            {"field": "status", "value": "path found"},
            {"field": "hop_count", "value": max(len(path_nodes) - 1, 0)},
            {"field": "path", "value": labeled_path},
        ]
    )


def build_component_frame(graph: nx.Graph, limit: int = 8) -> pd.DataFrame:
    """Summarize connected components for fast structure inspection."""

    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["component", "size", "share_of_nodes"])

    component_rows = []
    total_nodes = graph.number_of_nodes()
    for index, component_nodes in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True),
        start=1,
    ):
        size = len(component_nodes)
        component_rows.append(
            {
                "component": f"component_{index}",
                "size": size,
                "share_of_nodes": round(size / total_nodes, 3),
            }
        )

    return pd.DataFrame(component_rows).head(limit).reset_index(drop=True)


def build_graph_health_metrics(graph: nx.Graph) -> dict[str, float]:
    """Compute a few high-signal structural metrics for the current graph slice."""

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    if node_count == 0:
        return {
            "node_count": 0,
            "edge_count": 0,
            "component_count": 0,
            "density": 0.0,
            "avg_degree": 0.0,
        }

    avg_degree = round((2 * edge_count) / node_count, 2)
    density = round(nx.density(graph), 4) if node_count > 1 else 0.0
    component_count = nx.number_connected_components(graph)
    return {
        "node_count": node_count,
        "edge_count": edge_count,
        "component_count": component_count,
        "density": density,
        "avg_degree": avg_degree,
    }


def build_node_detail_frame(graph: nx.Graph, node_id: str) -> pd.DataFrame:
    """Convert one node's attributes into a compact inspector table."""

    if node_id not in graph:
        return pd.DataFrame(columns=["field", "value"])

    node_data = graph.nodes[node_id]
    rows = [{"field": "node_id", "value": node_id}]
    for key, value in sorted(node_data.items()):
        rows.append({"field": key, "value": value})
    rows.append({"field": "degree", "value": graph.degree(node_id)})
    return pd.DataFrame(rows)


def build_neighbor_frame(graph: nx.Graph, node_id: str) -> pd.DataFrame:
    """Build one readable neighbor table for the node inspector panel."""

    if node_id not in graph:
        return pd.DataFrame(columns=["neighbor", "node_type", "edge_type"])

    rows: list[dict[str, Any]] = []
    for neighbor_id in graph.neighbors(node_id):
        neighbor_data = graph.nodes[neighbor_id]
        edge_data = graph.get_edge_data(node_id, neighbor_id) or {}
        rows.append(
            {
                "neighbor": neighbor_data.get("label") or neighbor_id,
                "node_type": neighbor_data.get("node_type", "unknown"),
                "edge_type": edge_data.get("edge_type", "unknown"),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["neighbor", "node_type", "edge_type"])

    return (
        pd.DataFrame(rows)
        .sort_values(["node_type", "neighbor"], ascending=[True, True])
        .reset_index(drop=True)
    )


def build_pyvis_html(
    graph: nx.Graph,
    physics_enabled: bool,
    show_physics_controls: bool,
    height_px: int,
    style_profile: dict[str, Any],
    selected_node_id: str | None = None,
    path_result: dict[str, Any] | None = None,
) -> str:
    """Convert a NetworkX graph into draggable Pyvis HTML."""

    if Network is None:
        raise RuntimeError(
            "Pyvis is not installed. Run 'pip install -r requirements.txt'."
        )

    network = Network(
        height=f"{height_px}px",
        width="100%",
        directed=False,
        notebook=False,
        neighborhood_highlight=True,
        select_menu=True,
        filter_menu=True,
        bgcolor=style_profile["network_bg"],
        font_color=style_profile["network_font"],
        cdn_resources="in_line",
    )

    path_result = path_result or {"path_nodes": [], "path_edges": [], "status": "missing"}
    path_nodes = set(path_result.get("path_nodes") or [])
    path_edges = {
        tuple(edge)
        for edge in (path_result.get("path_edges") or [])
    }
    undirected_path_edges = path_edges | {(target, source) for source, target in path_edges}
    search_neighbor_ids = (
        set(graph.neighbors(selected_node_id))
        if selected_node_id and selected_node_id in graph
        else set()
    )
    has_highlight_context = bool(selected_node_id or path_nodes)

    # Translate the NetworkX graph into Pyvis nodes so the browser can render a
    # draggable physics-based network.
    for node_id, node_data in graph.nodes(data=True):
        degree = graph.degree(node_id)
        node_type = str(node_data.get("node_type", "unknown"))
        base_color = node_color(node_type, style_profile=style_profile)
        color_config: str | dict[str, Any] = base_color
        border_width = 1
        size = node_size(node_data, degree)

        if node_id in path_nodes:
            color_config = {
                "background": base_color,
                "border": "#DC2626",
                "highlight": {"background": base_color, "border": "#991B1B"},
            }
            border_width = 4
            size += 8
        elif selected_node_id and node_id == selected_node_id:
            color_config = {
                "background": base_color,
                "border": "#0F172A",
                "highlight": {"background": base_color, "border": "#020617"},
            }
            border_width = 4
            size += 6
        elif node_id in search_neighbor_ids:
            color_config = {
                "background": base_color,
                "border": "#F59E0B",
                "highlight": {"background": base_color, "border": "#D97706"},
            }
            border_width = 3
            size += 3
        elif has_highlight_context:
            color_config = hex_to_rgba(base_color, 0.22)

        network.add_node(
            node_id,
            label=str(node_data.get("label") or node_id),
            title=build_node_title(node_id, node_data, degree),
            color=color_config,
            size=size,
            borderWidth=border_width,
        )

    for source, target, edge_data in graph.edges(data=True):
        base_width = edge_width(edge_data)
        edge_style_color = edge_color(edge_data, style_profile=style_profile)
        edge_width_value = base_width
        if (source, target) in undirected_path_edges:
            edge_style_color = "#DC2626"
            edge_width_value = max(base_width + 2.5, 4.0)
        elif selected_node_id and selected_node_id in {source, target}:
            edge_style_color = "#0F172A"
            edge_width_value = max(base_width + 1.0, 2.5)
        elif has_highlight_context:
            edge_style_color = hex_to_rgba(edge_style_color, 0.18)

        network.add_edge(
            source,
            target,
            title=build_edge_title(edge_data),
            value=edge_width_value,
            width=edge_width_value,
            color=edge_style_color,
        )

    # Use the built-in physics helpers instead of overriding the full options
    # object, which keeps the optional control panel compatible with Pyvis.
    network.barnes_hut()
    network.toggle_physics(physics_enabled)
    network.options.interaction.hover = True
    network.options.interaction.navigationButtons = True
    network.options.interaction.keyboard = True
    network.options.edges.smooth = False
    # Some Streamlit Cloud / pyvis combinations expose parts of the options
    # tree as plain dict-like objects rather than attribute containers. Keep
    # the graph functional there instead of crashing on purely visual extras.
    try:
        network.options.nodes.borderWidth = 1
        network.options.nodes.shadow = True
        network.options.edges.selectionWidth = 2
        network.options.edges.hoverWidth = 1.2
    except AttributeError:
        pass

    if show_physics_controls:
        network.show_buttons(filter_=["physics"])

    return network.generate_html(notebook=False)


def counter_to_frame(counter_data) -> pd.DataFrame:
    """Convert counter-like objects into display tables."""

    rows = [
        {"type": item_type, "count": count}
        for item_type, count in sorted(counter_data.items())
    ]
    return pd.DataFrame(rows)


def inject_graph_app_styles(style_profile: dict[str, Any]) -> None:
    """Apply a lightweight explorer theme to the draggable graph app."""

    if st is None:
        return

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                radial-gradient(circle at top left, {style_profile["page_radial_left"]}, transparent 24%),
                radial-gradient(circle at top right, {style_profile["page_radial_right"]}, transparent 26%),
                linear-gradient(180deg, {style_profile["page_base_start"]} 0%, {style_profile["page_base_end"]} 100%);
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {style_profile["sidebar_start"]} 0%, {style_profile["sidebar_end"]} 100%);
            border-right: 1px solid {style_profile["metric_border"]};
        }}
        section[data-testid="stSidebar"] * {{
            color: #E5EEF8;
        }}
        div[data-testid="stMetric"] {{
            background: {style_profile["metric_bg"]};
            border: 1px solid {style_profile["metric_border"]};
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
        }}
        .graph-hero {{
            padding: 1.3rem 1.45rem;
            border-radius: 24px;
            color: white;
            background: linear-gradient(135deg, {style_profile["hero_start"]} 0%, {style_profile["hero_end"]} 100%);
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.16);
            margin-bottom: 1rem;
        }}
        .graph-hero h2 {{
            margin: 0 0 0.35rem 0;
            font-size: 2rem;
        }}
        .graph-hero p {{
            margin: 0;
            color: rgba(255, 255, 255, 0.92);
            line-height: 1.55;
        }}
        .graph-card {{
            background: {style_profile["card_bg"]};
            border: 1px solid {style_profile["card_border"]};
            border-radius: 18px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
        }}
        .graph-card h3 {{
            margin-top: 0;
            margin-bottom: 0.25rem;
            font-size: 1.05rem;
        }}
        .legend-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 0.75rem;
        }}
        .legend-card {{
            background: {style_profile["legend_bg"]};
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            border: 1px solid {style_profile["card_border"]};
        }}
        .legend-chip {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 999px;
            margin-right: 0.45rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def build_graph_type_frame(graph: nx.Graph, item: str) -> pd.DataFrame:
    """Summarize filtered node or edge counts by type for display panels."""

    if item == "node":
        counts: dict[str, int] = {}
        for _, node_data in graph.nodes(data=True):
            node_type = str(node_data.get("node_type", "unknown"))
            counts[node_type] = counts.get(node_type, 0) + 1
        return counter_to_frame(counts)

    counts = {}
    for _, _, edge_data in graph.edges(data=True):
        edge_type = str(edge_data.get("edge_type", "unknown"))
        counts[edge_type] = counts.get(edge_type, 0) + 1
    return counter_to_frame(counts)


def build_top_degree_frame(graph: nx.Graph, limit: int = 8) -> pd.DataFrame:
    """Build one small degree leaderboard for the filtered graph."""

    rows: list[dict[str, Any]] = []
    for node_id, node_data in graph.nodes(data=True):
        rows.append(
            {
                "label": node_data.get("label") or node_id,
                "node_type": node_data.get("node_type", "unknown"),
                "degree": graph.degree(node_id),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "node_type", "degree"])

    return (
        pd.DataFrame(rows)
        .sort_values(["degree", "node_type", "label"], ascending=[False, True, True])
        .head(limit)
        .reset_index(drop=True)
    )


def build_graph_control_frame(
    remove_isolates: bool,
    physics_enabled: bool,
    show_physics_controls: bool,
    height_px: int,
) -> pd.DataFrame:
    """Format display controls into one small user-facing table."""

    rows = [
        {
            "setting": "Remove isolated nodes",
            "value": "Yes" if remove_isolates else "No",
        },
        {
            "setting": "Physics enabled",
            "value": "Yes" if physics_enabled else "No",
        },
        {
            "setting": "Physics controls shown",
            "value": "Yes" if show_physics_controls else "No",
        },
        {
            "setting": "Graph height",
            "value": f"{height_px}px",
        },
    ]
    return pd.DataFrame(rows)


def render_legend_cards(style_profile: dict[str, Any]) -> None:
    """Render graph legend cards without relying on one large HTML blob."""

    legend_columns = st.columns(3)
    for column, node_type, accent in zip(
        legend_columns,
        ["stock", "sector", "topic"],
        style_profile["legend_accents"],
    ):
        with column:
            st.markdown(
                f"""
                <div class="legend-card" style="border-top: 4px solid {accent};">
                    <div><span class="legend-chip" style="background:{node_color(node_type, style_profile=style_profile)};"></span>{node_type}</div>
                    <div style="margin-top:0.45rem;color:#475569;font-size:0.9rem;">
                        color code for {node_type} nodes
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def run_app(*, configure_page: bool = True, embedded: bool = False) -> None:
    """Run the interactive graph visualization app."""

    if st is None or components is None:
        raise RuntimeError(
            "Streamlit is not installed. Run 'pip install -r requirements.txt'."
        )

    selected_style_profile = get_style_profile(DEFAULT_STYLE_PROFILE)

    if configure_page:
        st.set_page_config(page_title="Interactive Market Graph", layout="wide")
    inject_graph_app_styles(selected_style_profile)
    st.title("Interactive Market Graph Explorer")
    st.caption(
        "Graph browser built from the local project data. "
        "Use presets, focus mode, and the inspector panel to study structure, "
        "clusters, bridge nodes, and topic exposure."
    )

    with st.sidebar:
        st.header("Explorer Setup")
        style_options = build_style_profile_options()
        style_profile_name = st.selectbox(
            "Visual Style",
            options=style_options,
            index=style_options.index(DEFAULT_STYLE_PROFILE),
            help=(
                "Switch among three higher-end graph presentation styles: a "
                "product explorer shell, a cleaner canvas-driven browser, or a "
                "softer systems-map treatment."
            ),
        )
        selected_style_profile = get_style_profile(style_profile_name)
        ticker_text = st.text_input(
            "Ticker Filter",
            value=",".join(DEFAULT_GRAPH_TICKERS),
            help=(
                "Optional comma-separated tickers to limit the local graph. "
                "The default uses a smaller, cleaner multi-sector graph slice."
            ),
        )
        news_file = st.text_input("News File", value="merged_seed_news.json")
        correlation_threshold = st.slider(
            "Correlation Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
        )
        top_k_value = st.number_input(
            "Top-K Stock Neighbors",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
            help="Set to 0 to disable top-k mode.",
        )
        topic_weight = st.selectbox(
            "Stock-Topic Weight",
            options=[
                "article_count",
                "avg_topic_relevance",
                "avg_ticker_relevance",
                "avg_ticker_sentiment",
                "avg_overall_sentiment",
            ],
            index=0,
        )

        st.header("Explorer Mode")
        visual_preset = st.selectbox(
            "Visual Preset",
            options=list(VISUAL_PRESETS) + ["Custom"],
            index=0,
            help=(
                "Inspired by graph tools like Kumu and Cytoscape: presets give you "
                "a faster way to switch between broad market structure and a more "
                "targeted slice such as sector bridges or topic exposure."
            ),
        )
        if visual_preset == "Custom":
            included_node_types = set(
                st.multiselect(
                    "Node Types",
                    options=["stock", "sector", "topic"],
                    default=["stock", "sector", "topic"],
                )
            )
            included_edge_types = set(
                st.multiselect(
                    "Edge Types",
                    options=[
                        "stock_stock",
                        "stock_sector",
                        "stock_topic",
                        "sector_sector",
                    ],
                    default=[
                        "stock_stock",
                        "stock_sector",
                        "stock_topic",
                        "sector_sector",
                    ],
                )
            )
            preset_description = (
                "Custom mode lets you pick the exact node and edge types shown "
                "in the graph."
            )
        else:
            preset_config = VISUAL_PRESETS[visual_preset]
            included_node_types = set(preset_config["node_types"])
            included_edge_types = set(preset_config["edge_types"])
            preset_description = str(preset_config["description"])
            st.caption(preset_description)

        st.header("Display Controls")
        remove_isolates = st.checkbox("Remove Isolated Nodes", value=True)
        physics_enabled = st.checkbox("Enable Physics", value=True)
        show_physics_controls = st.checkbox("Show Physics Controls", value=True)
        height_px = st.slider("Graph Height", min_value=500, max_value=1100, value=760)

        st.caption(selected_style_profile["profile_summary"])

    inject_graph_app_styles(selected_style_profile)

    if embedded:
        st.caption(
            "Use the sidebar workspace switch to return to the analysis dashboard "
            "for stock, sector, path, and LLM impact queries."
        )
    else:
        st.info(
            "For structured stock / sector / path queries, use the separate site with "
            "`streamlit run query_app.py`."
        )

    tickers = parse_ticker_text(ticker_text)
    top_k = normalize_top_k(int(top_k_value))

    if not included_node_types:
        st.error("Select at least one node type.")
        st.stop()
    if not included_edge_types:
        st.error("Select at least one edge type.")
        st.stop()

    try:
        with st.spinner("Building local graph..."):
            # Keep this app on the exact same graph-building pipeline as the
            # query app; only the presentation layer changes here.
            analyzer, summary, context = load_analysis_bundle(
                tickers=tuple(tickers) if tickers is not None else None,
                news_file=news_file,
                correlation_threshold=correlation_threshold,
                top_k=top_k,
                topic_weight=topic_weight,
            )
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    base_filtered_graph = filter_graph(
        analyzer.graph,
        included_node_types=included_node_types,
        included_edge_types=included_edge_types,
        remove_isolates=remove_isolates,
    )

    focus_option_map = build_focus_node_option_map(base_filtered_graph)
    with st.sidebar:
        st.header("Focus View")
        selected_focus_label = st.selectbox(
            "Focus Node",
            options=["None"] + list(focus_option_map.keys()),
            index=0,
            help=(
                "Limit the graph to one local neighborhood. This mirrors the "
                "focus pattern used by tools like Kumu when exploring dense maps."
            ),
        )
        focus_hops = st.slider(
            "Focus Distance",
            min_value=1,
            max_value=3,
            value=1,
            disabled=selected_focus_label == "None",
            help="How many graph hops away from the focus node should stay visible.",
        )
        inspector_default_index = 0
        if selected_focus_label != "None":
            inspector_default_index = (
                ["None"] + list(focus_option_map.keys())
            ).index(selected_focus_label)
        selected_inspector_label = st.selectbox(
            "Inspector Node",
            options=["None"] + list(focus_option_map.keys()),
            index=inspector_default_index,
            help="Choose one node for a detail panel and neighbor table.",
        )
        st.header("Search And Path")
        selected_search_label = st.selectbox(
            "Search / Highlight Node",
            options=["None"] + list(focus_option_map.keys()),
            index=0,
            help="Highlight one node and its immediate neighbors in the graph.",
        )
        selected_path_source_label = st.selectbox(
            "Path Start",
            options=["None"] + list(focus_option_map.keys()),
            index=0,
            help="Pick the first endpoint for shortest-path highlighting.",
        )
        selected_path_target_label = st.selectbox(
            "Path End",
            options=["None"] + list(focus_option_map.keys()),
            index=0,
            help="Pick the second endpoint for shortest-path highlighting.",
        )

    focus_node_id = (
        focus_option_map[selected_focus_label]
        if selected_focus_label != "None"
        else None
    )
    filtered_graph = apply_focus_filter(
        base_filtered_graph,
        focus_node_id=focus_node_id,
        max_distance=focus_hops,
    )
    inspector_node_id = None
    if selected_inspector_label != "None":
        inspector_node_id = focus_option_map.get(selected_inspector_label)
    if inspector_node_id not in filtered_graph:
        inspector_node_id = next(iter(filtered_graph.nodes), None)
    search_node_id = (
        focus_option_map[selected_search_label]
        if selected_search_label != "None"
        else None
    )
    path_source_id = (
        focus_option_map[selected_path_source_label]
        if selected_path_source_label != "None"
        else None
    )
    path_target_id = (
        focus_option_map[selected_path_target_label]
        if selected_path_target_label != "None"
        else None
    )
    path_result = build_shortest_path_result(
        filtered_graph,
        source_node_id=path_source_id,
        target_node_id=path_target_id,
    )
    path_summary_frame = build_path_summary_frame(filtered_graph, path_result)

    st.markdown(
        """
        <div class="graph-hero">
            <h2>{hero_title}</h2>
            <p>
                {hero_body}
            </p>
        </div>
        """.format(
            hero_title=selected_style_profile["hero_title"],
            hero_body=selected_style_profile["hero_body"],
        ),
        unsafe_allow_html=True,
    )

    health_metrics = build_graph_health_metrics(filtered_graph)
    metric_columns = st.columns(6)
    metric_columns[0].metric("Price Tables", context["price_table_count"])
    metric_columns[1].metric("Articles", context["article_count"])
    metric_columns[2].metric("Nodes", health_metrics["node_count"])
    metric_columns[3].metric("Edges", health_metrics["edge_count"])
    metric_columns[4].metric("Components", health_metrics["component_count"])
    metric_columns[5].metric("Avg Degree", health_metrics["avg_degree"])

    node_type_frame = build_graph_type_frame(filtered_graph, item="node")
    edge_type_frame = build_graph_type_frame(filtered_graph, item="edge")
    top_degree_frame = build_top_degree_frame(filtered_graph)
    if filtered_graph.number_of_nodes() == 0:
        st.warning("No nodes remain after filtering. Adjust the sidebar filters.")
        return

    component_frame = build_component_frame(filtered_graph)
    node_detail_frame = (
        build_node_detail_frame(filtered_graph, inspector_node_id)
        if inspector_node_id
        else pd.DataFrame(columns=["field", "value"])
    )
    neighbor_frame = (
        build_neighbor_frame(filtered_graph, inspector_node_id)
        if inspector_node_id
        else pd.DataFrame(columns=["neighbor", "node_type", "edge_type"])
    )

    overview_columns = st.columns([1.25, 1.0])
    with overview_columns[0]:
        st.markdown(
            f"""
                <div class="graph-card">
                    <h3>Current Explorer Mode</h3>
                    <p><strong>{visual_preset}</strong></p>
                    <p>{preset_description}</p>
                    <p><strong>Style:</strong> {style_profile_name}</p>
                    <p><strong>Graph Default:</strong> cleaner starter slice, threshold {correlation_threshold:.2f}, top-k {top_k}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with overview_columns[1]:
        st.markdown(
            """
            <div class="graph-card">
                <h3>Visual Encoding</h3>
                <p>
                    Node color encodes node type, edge color encodes edge type,
                    and node size still scales with local prominence.
                </p>
                <p>{profile_summary}</p>
            </div>
            """.format(profile_summary=selected_style_profile["profile_summary"]),
            unsafe_allow_html=True,
        )

    canvas_tab, inspector_tab, structure_tab = st.tabs(
        ["Canvas", "Inspector", "Structure"]
    )

    with canvas_tab:
        html = build_pyvis_html(
            filtered_graph,
            physics_enabled=physics_enabled,
            show_physics_controls=show_physics_controls,
            height_px=height_px,
            style_profile=selected_style_profile,
            selected_node_id=search_node_id,
            path_result=path_result,
        )
        st.caption(selected_style_profile["canvas_caption"])
        if search_node_id:
            st.caption(
                f"Search highlight active: {filtered_graph.nodes[search_node_id].get('label') or search_node_id}"
            )
        if path_result["status"] == "ok":
            st.caption(
                f"Shortest path active with {len(path_result['path_nodes']) - 1} hops."
            )
        components.html(html, height=height_px + 50, scrolling=True)

    with inspector_tab:
        top_left, top_mid, top_right = st.columns([1.0, 1.0, 1.35])
        with top_left:
            st.caption("Inspector Node Details")
            st.dataframe(node_detail_frame, use_container_width=True, hide_index=True)
        with top_mid:
            st.caption("Shortest Path Summary")
            st.dataframe(path_summary_frame, use_container_width=True, hide_index=True)
        with top_right:
            st.caption("Neighbor Table")
            st.dataframe(neighbor_frame, use_container_width=True, hide_index=True)

    with structure_tab:
        details_column, legend_column = st.columns([3, 2])
        with details_column:
            st.markdown(
                """
                <div class="graph-card">
                    <h3>Filtered Graph Composition</h3>
                    <p>
                        This panel shows what remains after the current preset,
                        type filters, and focus-node restriction.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            left_inner, right_inner = st.columns(2)
            with left_inner:
                st.caption("Filtered Node Types")
                st.dataframe(node_type_frame, use_container_width=True, hide_index=True)
            with right_inner:
                st.caption("Filtered Edge Types")
                st.dataframe(edge_type_frame, use_container_width=True, hide_index=True)

            lower_left, lower_right = st.columns(2)
            with lower_left:
                st.caption("Top Nodes by Degree")
                st.dataframe(top_degree_frame, use_container_width=True, hide_index=True)
            with lower_right:
                st.caption("Connected Components")
                st.dataframe(component_frame, use_container_width=True, hide_index=True)

        with legend_column:
            st.markdown(
                """
                <div class="graph-card">
                    <h3>Legend And Controls</h3>
                    <p>
                        This view uses the selected style profile to tune the graph
                        shell, legend accents, and overall framing around the same
                        underlying network data.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            render_legend_cards(selected_style_profile)
            st.caption("Display Controls")
            st.dataframe(
                build_graph_control_frame(
                    remove_isolates=remove_isolates,
                    physics_enabled=physics_enabled,
                    show_physics_controls=show_physics_controls,
                    height_px=height_px,
                ),
                use_container_width=True,
                hide_index=True,
            )


if __name__ == "__main__":
    run_app()
