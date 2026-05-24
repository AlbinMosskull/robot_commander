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

## Verifying movement end-to-end

Start the server in the background, record the starting position, send a waypoint, wait for the agent to move, then assert the position has changed:

```
uv run agent-server --simulate &
uv run agent-cmd --simulate status          # note starting position
uv run agent-cmd --simulate checkpoint 1.5 2.0
sleep 5
uv run agent-cmd --simulate status          # confirm position has moved to (1.5, 2.0). If not there yet, wait a few more seconds and check again.
kill %1
```

If the agent successfully moves to the target position, it is likely that the system works, though depending on the experiment you are running there may be verfication needed for for example sensor state.

