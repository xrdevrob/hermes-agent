"""Context graph generator for Hermes session history.

Builds a lightweight knowledge/context graph from SessionDB data and exports
an interactive HTML visualization for sharing and exploration.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "your", "have", "will",
    "what", "when", "where", "which", "would", "there", "their", "about", "into",
    "just", "than", "then", "them", "they", "were", "been", "also", "only", "more",
    "some", "such", "could", "should", "very", "much", "does", "dont", "cant", "wont",
    "you", "are", "was", "not", "but", "too", "how", "why", "can", "our", "out",
    "its", "it", "to", "of", "in", "on", "at", "as", "is", "be", "or", "an", "a",
    "we", "i", "me", "my", "us", "he", "she", "his", "her", "if", "by", "do", "did",
    "using", "use", "used", "make", "made", "get", "got", "need", "needs", "want",
    "please", "thanks", "thank", "yes", "no", "okay", "ok", "sure", "here", "there",
}

_WORD_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_-]{2,}")


@dataclass
class GraphBuildResult:
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    stats: dict[str, Any]


class ContextGraphEngine:
    """Generate context graph data + HTML from a SessionDB instance."""

    def __init__(self, db):
        self.db = db
        self._conn = db._conn

    def generate(
        self,
        days: int = 30,
        source: str | None = None,
        max_sessions: int = 120,
        topics_per_session: int = 8,
        max_topic_edges: int = 200,
    ) -> GraphBuildResult:
        cutoff = time.time() - (days * 86400)
        sessions = self._get_sessions(cutoff=cutoff, source=source, limit=max_sessions)

        if not sessions:
            return GraphBuildResult(nodes=[], edges=[], stats={
                "days": days,
                "source": source,
                "sessions": 0,
                "topics": 0,
                "tools": 0,
                "edges": 0,
            })

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []

        session_topic_terms: dict[str, list[str]] = {}
        global_topic_counts: Counter[str] = Counter()
        topic_session_weight: dict[tuple[str, str], int] = defaultdict(int)
        topic_pair_weights: dict[tuple[str, str], int] = defaultdict(int)
        tool_session_weight: dict[tuple[str, str], int] = defaultdict(int)
        session_tool_totals: Counter[str] = Counter()

        # Session nodes first
        for s in sessions:
            sid = s["id"]
            label = self._session_label(s)
            msg_count = int(s.get("message_count") or 0)
            nodes.append({
                "id": f"session:{sid}",
                "label": label,
                "type": "session",
                "size": min(18 + (msg_count // 6), 36),
                "source": s.get("source") or "unknown",
                "session_id": sid,
                "messages": msg_count,
                "title": s.get("title") or "",
            })

            messages = self.db.get_messages(sid)
            tool_counts = self._extract_tool_counts(messages)
            term_counts = self._extract_term_counts(messages)
            top_terms = [term for term, _ in term_counts.most_common(topics_per_session)]
            session_topic_terms[sid] = top_terms

            for term in top_terms:
                global_topic_counts[term] += 1
                topic_session_weight[(sid, term)] += max(1, term_counts[term])

            for tool_name, count in tool_counts.items():
                tool_session_weight[(sid, tool_name)] += count
                session_tool_totals[tool_name] += count

            # Co-occurrence graph across top terms within a session
            for i in range(len(top_terms)):
                for j in range(i + 1, len(top_terms)):
                    a, b = sorted((top_terms[i], top_terms[j]))
                    topic_pair_weights[(a, b)] += 1

        # Topic nodes
        for topic, session_hits in global_topic_counts.most_common(120):
            nodes.append({
                "id": f"topic:{topic}",
                "label": topic,
                "type": "topic",
                "size": min(14 + session_hits * 2, 30),
                "hits": session_hits,
            })

        # Tool nodes
        for tool_name, calls in session_tool_totals.most_common(80):
            nodes.append({
                "id": f"tool:{tool_name}",
                "label": tool_name,
                "type": "tool",
                "size": min(14 + calls, 30),
                "calls": calls,
            })

        # Session -> topic edges
        for (sid, term), weight in topic_session_weight.items():
            edges.append({
                "from": f"session:{sid}",
                "to": f"topic:{term}",
                "type": "discusses",
                "weight": min(8, max(1, weight)),
            })

        # Session -> tool edges
        for (sid, tool_name), weight in tool_session_weight.items():
            edges.append({
                "from": f"session:{sid}",
                "to": f"tool:{tool_name}",
                "type": "uses",
                "weight": min(10, max(1, weight)),
            })

        # Topic <-> topic edges (limit for readability)
        pair_items = sorted(topic_pair_weights.items(), key=lambda x: x[1], reverse=True)[:max_topic_edges]
        for (a, b), weight in pair_items:
            if weight <= 0:
                continue
            edges.append({
                "from": f"topic:{a}",
                "to": f"topic:{b}",
                "type": "related",
                "weight": min(8, max(1, weight)),
            })

        stats = {
            "days": days,
            "source": source,
            "sessions": len(sessions),
            "topics": len(global_topic_counts),
            "tools": len(session_tool_totals),
            "edges": len(edges),
        }

        return GraphBuildResult(nodes=nodes, edges=edges, stats=stats)

    def export_html(
        self,
        graph: GraphBuildResult,
        output_path: str | Path | None = None,
        title: str = "Hermes Context Graph",
    ) -> Path:
        path = Path(output_path) if output_path else self._default_output_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "nodes": graph.nodes,
            "edges": graph.edges,
            "stats": graph.stats,
            "generated_at": time.time(),
            "title": title,
        }

        html = self._render_html(payload)
        path.write_text(html, encoding="utf-8")
        return path

    def format_summary(self, graph: GraphBuildResult, html_path: str | Path) -> str:
        s = graph.stats
        return (
            f"🕸️ **Context Graph Generated**\n"
            f"Sessions: {s.get('sessions', 0)} · Topics: {s.get('topics', 0)} · "
            f"Tools: {s.get('tools', 0)} · Edges: {s.get('edges', 0)}\n"
            f"File: `{html_path}`"
        )

    def _get_sessions(self, cutoff: float, source: str | None, limit: int) -> list[dict[str, Any]]:
        if source:
            cursor = self._conn.execute(
                """SELECT id, source, title, started_at, message_count
                   FROM sessions
                   WHERE started_at >= ? AND source = ?
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (cutoff, source, limit),
            )
        else:
            cursor = self._conn.execute(
                """SELECT id, source, title, started_at, message_count
                   FROM sessions
                   WHERE started_at >= ?
                   ORDER BY started_at DESC
                   LIMIT ?""",
                (cutoff, limit),
            )
        return [dict(r) for r in cursor.fetchall()]

    @staticmethod
    def _session_label(session: dict[str, Any]) -> str:
        title = (session.get("title") or "").strip()
        source = session.get("source") or "unknown"
        sid = session.get("id", "")
        short_id = sid[:8] if sid else "session"
        if title:
            return f"{title} ({source})"
        return f"{source}:{short_id}"

    @staticmethod
    def _extract_tool_counts(messages: list[dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for m in messages:
            tool_name = (m.get("tool_name") or "").strip()
            if tool_name:
                counts[tool_name] += 1

            tool_calls = m.get("tool_calls")
            if isinstance(tool_calls, list):
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name")
                        if name:
                            counts[str(name)] += 1
        return counts

    @staticmethod
    def _extract_term_counts(messages: list[dict[str, Any]]) -> Counter[str]:
        counts: Counter[str] = Counter()
        for m in messages:
            if m.get("role") not in {"user", "assistant"}:
                continue
            content = m.get("content") or ""
            if not content:
                continue
            for term in _WORD_RE.findall(content.lower()):
                if term in _STOPWORDS or term.isdigit() or len(term) < 3:
                    continue
                counts[term] += 1
        return counts

    @staticmethod
    def _default_output_path() -> Path:
        hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
        out_dir = hermes_home / "graphs"
        ts = time.strftime("%Y%m%d-%H%M%S")
        return out_dir / f"context-graph-{ts}.html"

    @staticmethod
    def _render_html(payload: dict[str, Any]) -> str:
        data_json = json.dumps(payload, ensure_ascii=False)
        template = """<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>Hermes Context Graph</title>
  <script src=\"https://unpkg.com/vis-network@9.1.9/dist/vis-network.min.js\"></script>
  <style>
    body { margin: 0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1020; color: #f4f7ff; }
    #top { padding: 12px 16px; border-bottom: 1px solid #2a3150; display: flex; gap: 20px; align-items: center; flex-wrap: wrap; }
    #graph { width: 100vw; height: calc(100vh - 68px); }
    .pill { background: #151d36; border: 1px solid #2a3150; border-radius: 999px; padding: 6px 10px; font-size: 12px; }
    .legend { display: flex; gap: 8px; align-items: center; }
  </style>
</head>
<body>
  <div id=\"top\">
    <strong id=\"title\">Hermes Context Graph</strong>
    <span class=\"pill\">sessions: <span id=\"sessions\"></span></span>
    <span class=\"pill\">topics: <span id=\"topics\"></span></span>
    <span class=\"pill\">tools: <span id=\"tools\"></span></span>
    <span class=\"pill\">edges: <span id=\"edges\"></span></span>
    <span class=\"legend\">session ● blue</span>
    <span class=\"legend\">topic ● purple</span>
    <span class=\"legend\">tool ● green</span>
  </div>
  <div id=\"graph\"></div>

  <script>
    const payload = __PAYLOAD_JSON__;
    const stats = payload.stats || {};
    document.getElementById('title').textContent = payload.title || 'Hermes Context Graph';
    document.getElementById('sessions').textContent = stats.sessions || 0;
    document.getElementById('topics').textContent = stats.topics || 0;
    document.getElementById('tools').textContent = stats.tools || 0;
    document.getElementById('edges').textContent = stats.edges || 0;

    const nodes = new vis.DataSet((payload.nodes || []).map(n => {
      let color = '#5b8def';
      if (n.type === 'topic') color = '#8b5cf6';
      if (n.type === 'tool') color = '#22c55e';
      return {
        id: n.id,
        label: n.label,
        value: n.size || 10,
        title: JSON.stringify(n, null, 2),
        color: { background: color, border: '#dbe4ff' },
        font: { color: '#f8fbff' },
      };
    }));

    const edges = new vis.DataSet((payload.edges || []).map(e => ({
      from: e.from,
      to: e.to,
      value: e.weight || 1,
      color: e.type === 'related' ? '#7b86b1' : '#55608f',
      smooth: e.type === 'related',
      title: `${e.type} (w=${e.weight || 1})`
    })));

    new vis.Network(
      document.getElementById('graph'),
      { nodes, edges },
      {
        interaction: { hover: true, navigationButtons: true, keyboard: true },
        nodes: { shape: 'dot', scaling: { min: 8, max: 36 } },
        edges: { width: 1, selectionWidth: 2, scaling: { min: 1, max: 6 } },
        physics: { stabilization: false, barnesHut: { gravitationalConstant: -25000 } },
      }
    );
  </script>
</body>
</html>
"""
        return template.replace("__PAYLOAD_JSON__", data_json)
