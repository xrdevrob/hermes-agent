# Hermes Hackathon Submission Plan

## Project
**Hermes Memory Map** — an interactive knowledge/context graph for Hermes Agent.

## One-liner
Turn Hermes session history into a visual graph so users can instantly see what topics they explored, what tools were used, and how context connects across sessions.

---

## Problem
Hermes users build lots of context over time, but it is hard to:
- understand what was learned,
- see recurring topics,
- trace which tools were used for which work,
- share progress in a visual, demo-friendly way.

## Solution
Add a native `/graph` command that generates an interactive HTML visualization from `state.db`.

Graph entities:
- **Session nodes**
- **Topic nodes** (keyword-derived)
- **Tool nodes**

Graph edges:
- session → topic (`discusses`)
- session → tool (`uses`)
- topic ↔ topic (`related`)

---

## MVP Scope (Hackathon)

### In-scope
- `/graph` command in CLI
- `/graph` command in gateway (Telegram/Discord)
- Interactive HTML export (vis-network)
- Filters: `--days`, `--source`
- Summary stats (sessions, topics, tools, edges)
- Basic test coverage

### Out-of-scope (post-hackathon)
- Full entity extraction (LLM or NER pipeline)
- Persistent graph DB (Neo4j)
- Temporal replay UI
- In-app embedded viewer

---

## Current Implementation Status
✅ Implemented:
- `agent/context_graph.py`
- CLI `/graph`
- Gateway `/graph`
- Telegram + Discord command registration
- Tests:
  - `tests/test_context_graph.py`
  - `tests/hermes_cli/test_commands.py` updated

✅ Targeted tests pass.

---

## Demo Plan (2–3 min)
1. Start from a real Hermes history.
2. Run `/graph --days 30`.
3. Open generated HTML and show node types/colors.
4. Zoom into a cluster (e.g., Discord + MCP + memory troubleshooting).
5. Highlight how this becomes a “memory map” of agent usage.
6. Close with: "Hermes can now visualize its context graph in one command."

---

## Submission Deliverables
- [ ] Code in fork (`xrdevrob/hermes-agent`)
- [ ] Short README section: "Context Graph"
- [ ] 2–3 minute demo video/GIF
- [ ] Before/after screenshots
- [ ] Final write-up (problem, solution, architecture, impact)

---

## Judging Narrative
- **Novelty:** Visual memory graph built directly from Hermes session history.
- **Utility:** Makes long-term context understandable and inspectable.
- **Polish:** One-command generation + interactive output.
- **Shareability:** Strong visual output for social demo clips.

---

## Stretch Goals (if time)
- Add evidence panel per node (sample messages/tool traces)
- Add `/graph --topic <keyword>` focus mode
- Add cluster labels and "top themes" summary
- Add optional PNG export for instant sharing
