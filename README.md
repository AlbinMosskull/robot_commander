# Robot Commander

robot_commander is a project in which a robot spider (the agent) is commanded around from a remote control station. The robot spider has limited processing power, and relies on the remote control station for heavy computation and instructions. The project uses cheap, simple sensors, but leverages machine learning to extend the information gained from these sensors.

**Key features**
- **Leverages ML to build a top-down view from a webcamera:** A webcamera watching the scene from an angle combined with DepthAnything and DETR+SAM generates a top-down "stencil map", which remote control can use to command the robot spider around
- **Remote Control provides an escape plan for completing the mission:** Obstacle mapping and path planning (built in rust) are run on remote control, but the agent holds an "escape plan", a way to return towards a safe position if it loses connection to the remote control station
- **Simple ultrasonic gives richer information thanks to DepthAnything:** DepthAnything combined with an ultrasonic sensor gives richer information at each reading from the sensor, for building the obstacle map

In this video, the operator in remote control is attempting to find a missing Yak, by sending commands to the agent to explore interesting regions.
<p align="center">
  <video src="https://github.com/user-attachments/assets/d94040a4-44ed-4174-9208-efb14564130f" controls width="100%" alt="Full system demo"></video>
</p>

## Algorithms

### Overhead Map

In order for remote control to be able to command the robot easily, we construct a top-down stencil map. That is, a map which is viewed from above (orthogonal to the ground), with key elements marked by their outline. In order to construct this map, we have access to a rgb web camera placed at an angle overlooking the scene but not from above. To produce the top-down stencil map, we employ the following algorithm.

1. The web camera gathers frames from its position
2. A depth image is produced from DepthAnything. The Metric DepthAnything is inexact, so instead two april tags placed in the scene are used as calibration references for the relative DepthAnything
3. By knowing the physical dimensions of the april tags, we can recover the exact depth to them. Using these two reference depth values, we can calibrate the depth image
4. With this depth image, we use RANSAC to detect the dominant plane, presumed to be the floor
5. We also run the color frames through a two-step instance segmentation pipeline: first DETR to find bounding boxes, then SAM to produce precise masks (this was found to be better than any off-the-shelf instance segmentation models)
6. For each detected object, we find the corresponding points in the point cloud generated from the depth image, then remove points close to the floor and filter for the largest connected cluster. This removes spurious points that leak into masks
7. The filtered points are projected onto the floor plane to produce an outline of each object's footprint
8. For each object, we also compute the region occluded from the camera's viewpoint
9. The resulting map is saved and served to remote control, which enables us to track objects in this map

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Map Building</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/5ffca650-2099-4952-a35c-acb583102747" autoplay loop muted playsinline width="100%" alt="Map building animation"></video>
      <br><br>
      <i>The different steps that we go through to build the map</i>
    </td>
    <td width="50%" align="center">
      <b>Overhead Map</b>
      <br><br>
      <img src="https://github.com/user-attachments/assets/985e76d3-6a78-4ebf-9b68-5ec14145c633" width="100%" alt="Static stencil map">
      <br><br>
      <i>The final stencil map, a top-down view of the scene, with the major objects</i>
    </td>
  </tr>
</table>

### Obstacle Mapping

In order to do intelligent path planning, we need to construct an obstacle map. The sensors available on the Adeept hardware platform are an ultrasonic distance sensor and an RGB camera. The ultrasonic sensor can give unreliable readings if the signal bounces and no direct reflection is received. The RGB camera can produce a depth image from DepthAnything, but without calibration this image is also untrustworthy in metric terms.

To work around this, the obstacle mapping employs two strategies depending on whether the sensor data can be trusted.

If RANSAC finds a well-defined plane in the point cloud from DepthAnything directly in front of the robot, that plane is a strong signal that the ultrasonic sensor had a clean, unobstructed reading. The ultrasonic distance is then used to calibrate the depth image, and the calibrated depth is used to update both the occupied and free regions of the obstacle map.

If no such plane is found, the ultrasonic reading is discarded. Instead, we conservatively mark only the free space we observe. Rays are extended to half the raw depth distance, claiming certainty of free space up to that point without committing to where obstacles are.

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Plane Matching</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/afbfe9c5-1d5e-4b0c-87ae-2a8c09a7bb19" autoplay loop muted playsinline width="100%" alt="Plane matching visualization"></video>
      <br><br>
      <i>When we find a plane in the point cloud in the sensor region, we calibrate the depth image and add obstacles for what we found</i>
    </td>
    <td width="50%" align="center">
      <b>Free Space</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/f90d33de-6c7e-4cf1-aeba-0647bf0510ee" autoplay loop muted playsinline width="100%" alt="Free space detection visualization"></video>
      <br><br>
      <i>If it is not clear what the ultrasonic sensor bounced off of, we update free space conservatively</i>
    </td>
  </tr>
</table>

### Path Planning

The user sets waypoints for the agent from Remote Control, and the system must generate a collision-free path to each one. The Adeept hardware platform exposes commands like "Forward", "Left", and "Right", where "Left" and "Right" rotate the robot in place. This means that the agent's kinematic model essentially is a unicycle: at any point it can either rotate or move straight.

Theta* is well-suited to this model. Unlike A*, it uses line-of-sight checks to shortcut paths between nodes, producing any-angle straight-line segments that map naturally onto the rotate-then-forward execution model. One modification, in that the planner never returns "no path found". If the goal is unreachable, it instead returns a path to the closest reachable node.

<table width="100%">
  <tr>
    <td width="50%" align="center">
      <b>Scenario 1</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/966b8fab-4e0c-4f58-bf90-c0d6d3189df0" autoplay loop muted playsinline width="100%" alt="Path planning scenario one"></video>
      <br><br>
      <i>Theta* planning in a solveable scenario</i>
    </td>
    <td width="50%" align="center">
      <b>Scenario 2</b>
      <br><br>
      <video src="https://github.com/user-attachments/assets/78e7725c-2e7d-4c52-b769-d6d4137cde6d" autoplay loop muted playsinline width="100%" alt="Path planning scenario two"></video>
      <br><br>
      <i>If the path planning cannot reach the target, we plan to a position as close as possible to the target</i>
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




