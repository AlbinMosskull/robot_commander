---
name: control-agent
description: Control the simulated robot agent — send waypoints and read state.
---

# Control Agent

## Setup (required first)

Start the agent server in a background terminal:
```
uv run agent-server --simulate
```

Then start the monitor to observe position and sensor state:
```
uv run agent-monitor --simulate
```

## Commands

Get current position (one-shot):
```
uv run agent-cmd --simulate status
```

Send a single waypoint:
```
uv run agent-cmd --simulate checkpoint <x> <y>
```

Send a multi-waypoint path:
```
uv run agent-cmd --simulate path <x1,y1> <x2,y2> …
```