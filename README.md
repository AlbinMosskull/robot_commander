# Robot Commander

robot_commander is a project in which a robot spider is commanded around from a remote control station. The robot spider has limited processing power, and relies on the remote control station for heavy computation and instructions. The project uses cheap, simple sensors, but leverages machine learning to extend the information gained from these sensors.

- A webcamera watching the scene from an angle combined with DepthAnything generates a top-down "stencil map", which remote control can use to command the robot spider around
- Obstacle mapping and path planning (built in rust) are run on remote control, but the agent maintains an "escape plan", a way to return towards a safe position if it loses connection to the remote control station
- DepthAnything combined with an ultrasonic sensor gives richer information at each reading from the sensor, for building the obstacle map


## Running
Start the agent (either on the edge computer, or on the remote control computer)
```bash
uv run agent-server
```

and then, on the remote control computer

```bash
uv run uv run remote-control
```

## Installation

### Desktop
Create a virtual environment and install remote control dependencies 

```bash
uv venv
source .venv/bin/activate
uv sync --extra remote_control
```
### On the edge computer (assumed raspberry pi)

1. Clone the repo
2. Install system dependencies (not available via pip)
    - ```bash
         sudo apt-get install i2c-tools python3-smbus libcap-dev
        ```
3. Create a virtual environment and install agent dependencies only 
    - ```bash
        uv venv
        source .venv/bin/activate
        uv sync --extra agent
        ```

## Calibration
**Calibration - Intrinsics**

The first step is to find the intrinsics of your camera.
1. Print a checkerboard. You can find these from OpenCV, and print onto an A4 paper.
2. Mount the checkboard on something with a hard back, such that it may not bend.
3. Use python/robot_commander/image_processing/run_capture_images.py to capture images of the checkerboard from different positions of the camera frame, and with different tilts. Go for about 15-20 images.
4. Run python/robot_commander/image_processing/calibrate_intrinsics.py to compute intrinsics. The reprojection error should ideally be under 1 pixel if everything works as expected.

**Calibration - depth**

Get two april tags, and put them in the scene. One should lay flat against the floor. And the other at a position with lower depth.


## Conventions
- Transforms should be named like camera_T_sensor and be 4x4




