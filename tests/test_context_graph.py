"""Tests for agent/context_graph.py."""

from pathlib import Path

from hermes_state import SessionDB
from agent.context_graph import ContextGraphEngine


def _make_db(tmp_path: Path) -> SessionDB:
    db = SessionDB(db_path=tmp_path / "graph_test.db")

    db.create_session("s1", source="cli", model="gpt-4o")
    db.append_message("s1", role="user", content="Optimize docker build caching for python service")
    db.append_message(
        "s1",
        role="assistant",
        content="I will inspect Dockerfile and run tests",
        tool_calls=[{"function": {"name": "search_files"}}],
    )
    db.append_message("s1", role="tool", content="hits", tool_name="search_files")

    db.create_session("s2", source="discord", model="gpt-4o-mini")
    db.append_message("s2", role="user", content="Need telegram webhook troubleshooting steps")
    db.append_message(
        "s2",
        role="assistant",
        content="Let's check logs and config",
        tool_calls=[{"function": {"name": "read_file"}}],
    )
    db.append_message("s2", role="tool", content="ok", tool_name="read_file")
    db._conn.commit()

    return db


def test_generate_graph_contains_session_topic_tool_nodes(tmp_path):
    db = _make_db(tmp_path)
    try:
        engine = ContextGraphEngine(db)
        graph = engine.generate(days=365)

        node_ids = {n["id"] for n in graph.nodes}
        assert "session:s1" in node_ids
        assert "session:s2" in node_ids
        assert any(n["id"].startswith("topic:") for n in graph.nodes)
        assert "tool:search_files" in node_ids
        assert "tool:read_file" in node_ids

        assert graph.stats["sessions"] == 2
        assert graph.stats["edges"] > 0
    finally:
        db.close()


def test_export_html_writes_interactive_file(tmp_path):
    db = _make_db(tmp_path)
    try:
        engine = ContextGraphEngine(db)
        graph = engine.generate(days=365)
        out = tmp_path / "context-graph.html"
        path = engine.export_html(graph, output_path=out)

        assert path.exists()
        html = path.read_text(encoding="utf-8")
        assert "cytoscape" in html
        assert "payload" in html
        assert "session:s1" in html
    finally:
        db.close()
