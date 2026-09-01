"""Tests for interactive graph visualization helpers."""

from __future__ import annotations

import networkx as nx

from interactive_graph_app import (
    apply_focus_filter,
    build_graph_control_frame,
    build_focus_node_option_map,
    build_graph_health_metrics,
    build_path_summary_frame,
    build_shortest_path_result,
    build_style_profile_options,
    build_graph_type_frame,
    build_edge_title,
    build_node_title,
    build_top_degree_frame,
    edge_width,
    edge_color,
    filter_graph,
    get_style_profile,
    node_color,
    node_size,
)


def build_sample_graph() -> nx.Graph:
    """Create a small mixed-type graph for helper tests."""

    graph = nx.Graph()
    graph.add_node("stock:AAPL", node_type="stock", label="AAPL", ticker="AAPL")
    graph.add_node(
        "sector:Technology",
        node_type="sector",
        label="Technology",
        sector="Technology",
        stock_count=1,
    )
    graph.add_node(
        "topic:technology",
        node_type="topic",
        label="technology",
        topic="technology",
        article_count=40,
    )
    graph.add_edge(
        "stock:AAPL",
        "sector:Technology",
        edge_type="stock_sector",
        weight=1.0,
    )
    graph.add_edge(
        "stock:AAPL",
        "topic:technology",
        edge_type="stock_topic",
        weight=40,
        article_count=40,
        avg_ticker_sentiment=0.2,
    )
    return graph


def test_filter_graph_keeps_only_requested_types() -> None:
    """Filtering should remove unselected node and edge types."""

    graph = build_sample_graph()

    filtered = filter_graph(
        graph,
        included_node_types={"stock", "sector"},
        included_edge_types={"stock_sector"},
        remove_isolates=True,
    )

    assert set(filtered.nodes()) == {"stock:AAPL", "sector:Technology"}
    assert filtered.number_of_edges() == 1


def test_node_helpers_return_stable_display_values() -> None:
    """Node styling helpers should produce deterministic display values."""

    graph = build_sample_graph()
    stock_data = graph.nodes["stock:AAPL"]

    assert node_color("stock") == "#2F6BFF"
    assert node_size(stock_data, degree=2) > 0
    assert "ticker: AAPL" in build_node_title("stock:AAPL", stock_data, degree=2)


def test_style_profiles_are_available_and_change_visual_tokens() -> None:
    """The graph explorer should expose the three supported style profiles."""

    style_options = build_style_profile_options()
    sigma_profile = get_style_profile("AWS + Sigma")
    fallback_profile = get_style_profile("does-not-exist")

    assert style_options == ["AWS + Bloom", "AWS + Sigma", "Kumu + Bloom"]
    assert node_color("topic", style_profile=sigma_profile) == "#7C3AED"
    assert fallback_profile["hero_title"] == "Graph Explorer Workbench"


def test_edge_helpers_include_metadata() -> None:
    """Edge helpers should surface important edge fields."""

    graph = build_sample_graph()
    edge_data = graph.get_edge_data("stock:AAPL", "topic:technology")
    bloom_profile = get_style_profile("AWS + Bloom")

    assert edge_width(edge_data) >= 1.0
    assert edge_color(edge_data, style_profile=bloom_profile) == "#D97706"
    assert "article_count: 40" in build_edge_title(edge_data)


def test_build_graph_type_frame_summarizes_filtered_graph() -> None:
    """Type summary helper should count nodes and edges by type."""

    graph = build_sample_graph()

    node_frame = build_graph_type_frame(graph, item="node")
    edge_frame = build_graph_type_frame(graph, item="edge")

    assert set(node_frame["type"]) == {"stock", "sector", "topic"}
    assert set(edge_frame["type"]) == {"stock_sector", "stock_topic"}


def test_build_top_degree_frame_orders_nodes_by_degree() -> None:
    """Top-degree helper should expose readable labels and degree values."""

    graph = build_sample_graph()
    frame = build_top_degree_frame(graph, limit=3)

    assert frame.loc[0, "label"] == "AAPL"
    assert frame.loc[0, "degree"] == 2


def test_build_graph_control_frame_formats_values_for_display() -> None:
    """Display-control helper should expose readable yes/no and size text."""

    frame = build_graph_control_frame(
        remove_isolates=True,
        physics_enabled=False,
        show_physics_controls=True,
        height_px=760,
    )

    assert frame.to_dict("records") == [
        {"setting": "Remove isolated nodes", "value": "Yes"},
        {"setting": "Physics enabled", "value": "No"},
        {"setting": "Physics controls shown", "value": "Yes"},
        {"setting": "Graph height", "value": "760px"},
    ]


def test_apply_focus_filter_keeps_local_neighborhood() -> None:
    """Focus filtering should reduce the graph to one node neighborhood."""

    graph = build_sample_graph()

    focused = apply_focus_filter(
        graph,
        focus_node_id="stock:AAPL",
        max_distance=1,
    )

    assert set(focused.nodes()) == {
        "stock:AAPL",
        "sector:Technology",
        "topic:technology",
    }


def test_focus_option_map_and_health_metrics_are_readable() -> None:
    """Explorer helpers should expose readable labels and stable metrics."""

    graph = build_sample_graph()

    option_map = build_focus_node_option_map(graph)
    metrics = build_graph_health_metrics(graph)

    assert "stock | AAPL | degree 2" in option_map
    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 2
    assert metrics["component_count"] == 1


def test_shortest_path_helpers_report_found_path() -> None:
    """Path helpers should build a readable path summary when a path exists."""

    graph = build_sample_graph()
    path_result = build_shortest_path_result(
        graph,
        source_node_id="sector:Technology",
        target_node_id="topic:technology",
    )
    summary_frame = build_path_summary_frame(graph, path_result)

    assert path_result["status"] == "ok"
    assert path_result["path_nodes"] == [
        "sector:Technology",
        "stock:AAPL",
        "topic:technology",
    ]
    assert summary_frame.loc[1, "value"] == 2


def test_shortest_path_helpers_report_missing_path() -> None:
    """Path helpers should stay readable when no path can be found."""

    graph = build_sample_graph()
    graph.add_node("topic:isolated", node_type="topic", label="isolated")

    path_result = build_shortest_path_result(
        graph,
        source_node_id="stock:AAPL",
        target_node_id="topic:isolated",
    )
    summary_frame = build_path_summary_frame(graph, path_result)

    assert path_result["status"] == "no-path"
    assert summary_frame.to_dict("records") == [
        {"field": "status", "value": "no path"}
    ]
