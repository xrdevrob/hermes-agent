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
        topics_per_session: int = 6,
        max_topic_edges: int = 120,
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

        # Topic nodes (cap for readability)
        for topic, session_hits in global_topic_counts.most_common(90):
            nodes.append({
                "id": f"topic:{topic}",
                "label": topic,
                "type": "topic",
                "size": min(14 + session_hits * 2, 28),
                "hits": session_hits,
            })

        # Tool nodes (cap for readability)
        for tool_name, calls in session_tool_totals.most_common(50):
            nodes.append({
                "id": f"tool:{tool_name}",
                "label": self._humanize_tool_name(tool_name),
                "type": "tool",
                "size": min(14 + calls, 28),
                "calls": calls,
                "tool_name": tool_name,
            })

        node_ids = {n["id"] for n in nodes}

        # Session -> topic edges
        for (sid, term), weight in topic_session_weight.items():
            src = f"session:{sid}"
            dst = f"topic:{term}"
            if src not in node_ids or dst not in node_ids:
                continue
            edges.append({
                "from": src,
                "to": dst,
                "type": "discusses",
                "weight": min(8, max(1, weight)),
            })

        # Session -> tool edges
        for (sid, tool_name), weight in tool_session_weight.items():
            src = f"session:{sid}"
            dst = f"tool:{tool_name}"
            if src not in node_ids or dst not in node_ids:
                continue
            edges.append({
                "from": src,
                "to": dst,
                "type": "uses",
                "weight": min(10, max(1, weight)),
            })

        # Topic <-> topic edges (limit and weight threshold for readability)
        pair_items = sorted(topic_pair_weights.items(), key=lambda x: x[1], reverse=True)[:max_topic_edges]
        for (a, b), weight in pair_items:
            if weight < 2:
                continue
            src = f"topic:{a}"
            dst = f"topic:{b}"
            if src not in node_ids or dst not in node_ids:
                continue
            edges.append({
                "from": src,
                "to": dst,
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
    def _humanize_tool_name(name: str) -> str:
        if not name:
            return name
        pretty = {
            "patch": "File Edit (patch)",
            "read_file": "Read File",
            "search_files": "Search Files",
            "write_file": "Write File",
            "terminal": "Shell Command",
            "browser_navigate": "Browser Navigate",
            "browser_snapshot": "Browser Snapshot",
            "browser_click": "Browser Click",
            "session_search": "Session Recall",
        }
        if name in pretty:
            return pretty[name]
        return name.replace("_", " ").strip().title()

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
  <script src=\"https://unpkg.com/cytoscape@3.30.2/dist/cytoscape.min.js\"></script>
  <script src=\"https://unpkg.com/layout-base@2.0.1/layout-base.js\"></script>
  <script src=\"https://unpkg.com/cose-base@2.2.0/cose-base.js\"></script>
  <script src=\"https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js\"></script>
  <style>
    body { margin: 0; font-family: Inter, -apple-system, Segoe UI, Roboto, sans-serif; background: #0b1020; color: #f4f7ff; }
    #top { padding: 12px 16px; border-bottom: 1px solid #2a3150; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
    #graph { width: calc(100vw - 300px); height: calc(100vh - 68px); display: inline-block; }
    #side { width: 300px; height: calc(100vh - 68px); display: inline-block; vertical-align: top; background:#0e152a; border-left:1px solid #2a3150; box-sizing:border-box; padding:12px; overflow:auto; }
    .pill { background: #151d36; border: 1px solid #2a3150; border-radius: 999px; padding: 6px 10px; font-size: 12px; }
    .legend { display: flex; gap: 8px; align-items: center; }
    .hint { color:#9fb0d9; font-size:11px; }
    #search { background: #0f1730; border: 1px solid #2a3150; color: #e8efff; border-radius: 8px; padding: 6px 8px; min-width: 170px; }
    #btn-fit { background: #1d2a4d; color: #e9f0ff; border: 1px solid #30406f; border-radius: 8px; padding: 6px 10px; cursor: pointer; }
  </style>
</head>
<body>
  <div id=\"top\">
    <strong id=\"title\">Hermes Context Graph</strong>
    <span class=\"pill\">sessions: <span id=\"sessions\"></span></span>
    <span class=\"pill\">topics: <span id=\"topics\"></span></span>
    <span class=\"pill\">tools: <span id=\"tools\"></span></span>
    <span class=\"pill\">edges: <span id=\"edges\"></span></span>
    <input id=\"search\" type=\"text\" placeholder=\"highlight node label...\" />
    <button id=\"btn-fit\">Fit graph</button>
    <label class=\"hint\"><input id=\"labels-all\" type=\"checkbox\" /> show more labels</label>
    <span class=\"legend\">session ● blue</span>
    <span class=\"legend\">topic ● purple</span>
    <span class=\"legend\">tool ● green</span>
    <span class=\"hint\">Node size = frequency / importance</span>
  </div>
  <div id=\"graph\"></div><div id=\"side\"><h3 style=\"margin-top:0\">Node details</h3><div id=\"details\" class=\"hint\">Click a node to inspect it.</div></div>

  <script>
    const payload = __PAYLOAD_JSON__;
    const stats = payload.stats || {};
    document.getElementById('title').textContent = payload.title || 'Hermes Context Graph';
    document.getElementById('sessions').textContent = stats.sessions || 0;
    document.getElementById('topics').textContent = stats.topics || 0;
    document.getElementById('tools').textContent = stats.tools || 0;
    document.getElementById('edges').textContent = stats.edges || 0;

    const nodeColor = (type) => {
      if (type === 'session') return '#5b8def';
      if (type === 'topic') return '#8b5cf6';
      if (type === 'tool') return '#22c55e';
      return '#9ca3af';
    };

    const elements = [];
    for (const n of (payload.nodes || [])) {
      const isBig = (n.size || 0) >= 18;
      const showLabel = n.type === 'session' || n.type === 'tool' || isBig;
      elements.push({
        data: {
          id: n.id,
          label: showLabel ? n.label : '',
          defaultLabel: showLabel ? n.label : '',
          fullLabel: n.label,
          type: n.type,
          size: n.size || 10,
          color: nodeColor(n.type),
          raw: JSON.stringify(n),
        }
      });
    }

    for (const e of (payload.edges || [])) {
      elements.push({
        data: {
          id: `${e.from}->${e.to}:${e.type}`,
          source: e.from,
          target: e.to,
          weight: e.weight || 1,
          rel: e.type || 'related'
        }
      });
    }

    const cy = cytoscape({
      container: document.getElementById('graph'),
      elements,
      style: [
        {
          selector: 'node',
          style: {
            'background-color': 'data(color)',
            'border-color': '#dbe4ff',
            'border-width': 1,
            'width': 'mapData(size, 8, 34, 10, 34)',
            'height': 'mapData(size, 8, 34, 10, 34)',
            'label': 'data(label)',
            'color': '#f5f8ff',
            'font-size': 11,
            'text-wrap': 'none',
            'text-max-width': 120,
            'text-outline-width': 2,
            'text-outline-color': '#0b1020',
            'text-margin-y': -12,
          }
        },
        {
          selector: 'edge',
          style: {
            'line-color': '#6070a5',
            'opacity': 0.38,
            'width': 'mapData(weight, 1, 8, 1, 4)',
            'curve-style': 'bezier'
          }
        },
        {
          selector: 'node:selected',
          style: {
            'border-width': 3,
            'border-color': '#ffffff',
            'z-index': 999
          }
        },
        {
          selector: '.faded',
          style: {
            'opacity': 0.12
          }
        },
        {
          selector: '.active',
          style: {
            'opacity': 1,
            'line-color': '#93a4df',
            'background-color': '#ffd166',
            'color': '#fff'
          }
        }
      ],
      layout: {
        name: 'fcose',
        quality: 'proof',
        randomize: true,
        animate: false,
        fit: true,
        padding: 40,
        nodeRepulsion: 8000,
        idealEdgeLength: 140,
        edgeElasticity: 0.3,
        gravity: 0.25,
        numIter: 2500,
        tile: true,
        packComponents: true,
      }
    });

    document.getElementById('btn-fit').addEventListener('click', () => cy.fit(undefined, 40));

    const searchEl = document.getElementById('search');
    searchEl.addEventListener('input', () => {
      const q = searchEl.value.trim().toLowerCase();
      cy.elements().removeClass('active').removeClass('faded');
      if (!q) return;
      const matches = cy.nodes().filter(n => (n.data('fullLabel') || '').toLowerCase().includes(q));
      const keep = matches.union(matches.connectedEdges()).union(matches.connectedEdges().connectedNodes());
      cy.elements().difference(keep).addClass('faded');
      keep.addClass('active');
    });

    const labelsAll = document.getElementById('labels-all');
    labelsAll.addEventListener('change', () => {
      const showAll = labelsAll.checked;
      cy.nodes().forEach(n => {
        n.data('label', showAll ? (n.data('fullLabel') || '') : (n.data('defaultLabel') || ''));
      });
    });

    const details = document.getElementById('details');
    cy.on('tap', 'node', (evt) => {
      const n = evt.target;
      let raw = {};
      try { raw = JSON.parse(n.data('raw') || '{}'); } catch (_) { raw = {}; }
      const neighbors = n.connectedEdges().connectedNodes().length - 1;
      const edgeCount = n.connectedEdges().length;
      details.innerHTML = `
        <div><strong>${n.data('fullLabel') || n.id()}</strong></div>
        <div class='hint'>type: ${n.data('type')} · size: ${n.data('size')} · edges: ${edgeCount} · neighbors: ${neighbors}</div>
        <hr style='border-color:#2a3150;border-width:1px 0 0 0;margin:8px 0'/>
        <pre style='white-space:pre-wrap;color:#d6e0ff;font-size:11px;line-height:1.35'>${JSON.stringify(raw, null, 2)}</pre>
      `;
    });
  </script>
</body>
</html>
"""
        return template.replace("__PAYLOAD_JSON__", data_json)
