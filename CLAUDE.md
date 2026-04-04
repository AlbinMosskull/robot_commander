# CLAUDE.md

## Project Overview
This is a project about controlling a robot agent, from a remote control station.

The agent does not do heavy computation by itself, and will instead rely on the remote control station for computations and plans.
The remote control station allows a user to set waypoints for the agent through a GUI, and runs path planning and occlusion mapping.

The stack is Python & Rust.

## Code style
Code style is very important for the repo.
Modules and functions should strive to have one singular clear purpose.
Comments should be used sparingly, and we should rely on good naming and small functions for readability.

Do:
- Break out new functions and files when the current one grows too large and is responsible for too much
- Add tests

Do not:
- Do not maintain backward compatability or only go half-way with refactors
- Do not add exessive comments that will not help a future reader

## Running
- ```uv run pytest path-to-module``` : for running python tests
- ```uv run pytest path-to-file``` : for running python
- ```cargo build``` : build rust code
- ```cargo test``` : run rust tests
- ```maturin develop``` : build rust bindings for python

