# Robot Commander

robot_commander is a project in which a robot spider (the agent) is commanded around from a remote control station. The robot spider has limited processing power, and relies on the remote control station for heavy computation and instructions. The project uses cheap, simple sensors, but leverages machine learning to extend the information gained from these sensors.

**Key features**
- **Leverages ML to build a top-down view from a webcamera:** A webcamera watching the scene from an angle combined with DepthAnything and YOLO+SAM generates a top-down "stencil map", which remote control can use to command the robot spider around
- **Remote Control provides an escape plan for completing the mission:** Obstacle mapping and path planning (built in rust) are run on remote control, but the agent holds an "escape plan", a way to return towards a safe position if it loses connection to the remote control station
- **Simple ultrasonic gives richer information thanks to DepthAnything:** DepthAnything combined with an ultrasonic sensor gives richer information at each reading from the sensor, for building the obstacle map

![Demo](media/full_run.mp4)

## Hardware setup
The project currently supports the Adeept Raspclaws Ultimate spider, and it is also possible to run fully in simulation.

For a proper hardware setup, you need to assemble the spider, and ensure you have remote access to your raspberry pi. You should also mount a webcamera such that is oversees the scene in which you want to control the robot spider, such as by mounting it to a chair.

## Installation

### Desktop
Create a virtual environment and install remote control dependencies 

1. Install python dependencies
    ```bash
        uv venv
        source .venv/bin/activate
        uv sync --extra remote_control
    ```
2. Compile the rust libraries for usage within the python code
    ```bash
        maturin develop
    ```



### On the edge computer (assumed raspberry pi)

1. Clone the repo
2. Install system dependencies (not available via pip)
    ```bash
         sudo apt-get install i2c-tools python3-smbus libcap-dev
    ```
3. Create a virtual environment with access to system site-packages (required for
   `libcamera` and `picamera2`, which are installed system-wide on Raspberry Pi OS
   and cannot be installed via pip)
    ```bash
        uv venv --system-site-packages
        source .venv/bin/activate
        uv sync --extra agent
    ```

## Calibration
**Calibration - Intrinsics**

The first step is to find the intrinsics of your camera.
1. Print a checkerboard. You can find these from OpenCV, and print onto an A4 paper.
2. Mount the checkboard on something with a hard back, such that it may not bend.
3. Use ```uv run calibrate-camera``` to capture images of the checkerboard from different positions of the camera frame, and with different tilts. Go for about 15-20 images.
4. Once you exit, the reprojection error will be computed. It should ideally be under 1 pixel if everything works as expected.

**Calibration - depth**

Get two april tags, and put them in the scene. One should lay flat against the floor. And the other at a position with lower depth. By doing this, the stencil map will be built properly.

## Running
Start the agent (either on the edge computer, or on the remote control computer)
```bash
uv run agent-server
```

and then, on the remote control computer

```bash
uv run remote-control
```


## Conventions
- Transforms should be named like camera_T_sensor and be 4x4




