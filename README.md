# Robot Commander

robot_commander is a project in which a robot spider (the agent) is commanded around from a remote control station. The robot spider has limited processing power, and relies on the remote control station for heavy computation and instructions. The project uses cheap, simple sensors, but leverages machine learning to extend the information gained from these sensors.

**Key features**
- **Leverages ML to build a top-down view from a webcamera:** A webcamera watching the scene from an angle combined with DepthAnything and YOLO+SAM generates a top-down "stencil map", which remote control can use to command the robot spider around
- **Remote Control provides an escape plan for completing the mission:** Obstacle mapping and path planning (built in rust) are run on remote control, but the agent holds an "escape plan", a way to return towards a safe position if it loses connection to the remote control station
- **Simple ultrasonic gives richer information thanks to DepthAnything:** DepthAnything combined with an ultrasonic sensor gives richer information at each reading from the sensor, for building the obstacle map

<p align="center">
  <video src="https://github.com/user-attachments/assets/d94040a4-44ed-4174-9208-efb14564130f" controls width="100%" alt="Full system demo"></video>
</p>

## Algorithms

### Overhead Map

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Map Building</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/5ffca650-2099-4952-a35c-acb583102747" autoplay loop muted playsinline width="100%" alt="Map building animation"></video>
    </td>
    <td width="50%" align="center">
      <b>Overhead Map</b>
      <br><br>
      <img src="https://github.com/user-attachments/assets/985e76d3-6a78-4ebf-9b68-5ec14145c633" width="100%" alt="Static stencil map">
    </td>
  </tr>
</table>

### Obstacle Mapping

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Plane Matching</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/afbfe9c5-1d5e-4b0c-87ae-2a8c09a7bb19" autoplay loop muted playsinline width="100%" alt="Plane matching visualization"></video>
    </td>
    <td width="50%" align="center">
      <b>Free Space</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/f90d33de-6c7e-4cf1-aeba-0647bf0510ee" autoplay loop muted playsinline width="100%" alt="Free space detection visualization"></video>
    </td>
  </tr>
</table>

### Path Planning

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Scenario 1</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/966b8fab-4e0c-4f58-bf90-c0d6d3189df0" autoplay loop muted playsinline width="100%" alt="Path planning scenario one"></video>
    </td>
    <td width="50%" align="center">
      <b>Scenario 2</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/78e7725c-2e7d-4c52-b769-d6d4137cde6d" autoplay loop muted playsinline width="100%" alt="Path planning scenario two"></video>
    </td>
  </tr>
</table>



## Hardware setup
The project currently supports the Adeept Raspclaws Ultimate spider, and it is also possible to run fully in simulation.

For a proper hardware setup, you need to assemble the spider, and ensure you have remote access to your raspberry pi. You should also mount a webcamera such that is oversees the scene in which you want to control the robot spider, such as by mounting it to a chair. You need to add an april tag to your spider, mounted on top of the camera.

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

The first step is to find the intrinsics of your cameras (overhead camera and agent camera).
1. Print a checkerboard. You can find these from OpenCV, and print onto an A4 paper.
2. Mount the checkboard on something with a hard back, such that it may not bend.
3. Use ```uv run calibrate-camera``` to capture images of the checkerboard from different positions of the camera frame, and with different tilts. Go for about 15-20 images.
4. Once you exit, the reprojection error will be computed. It should ideally be under 1 pixel if everything works as expected.

**Calibration - depth**

Get an additional april tag, apart from the one you have on the robot, and put it in the scene. Add them at points with different distances to your overhead camera. By doing this, the stencil map will be built properly.

## Running
Start the agent (either on the edge computer, or on the remote control computer)
```bash
uv run agent-server
```

and then, on the remote control computer

```bash
uv run remote-control
```

add a 

```bash
--simulate
```
flag to both of the scripts to run in simulation.

### Controls

| Control | Action |
|---------|--------|
| Left click on map | Set waypoint, no path planning |
| Shift + left drag on map | Set waypoint with heading, path planning |
| `S` | Scout, rotate in place to observe more |
| `P` | Enable payload, capture an image at the end of the current plan, store to payload |


## Conventions
- Transforms should be named like camera_T_sensor and be 4x4




