# Robot Commander

## Setup
**Installation**
```git clone git@github.com:adeept/Adeept_RaspClaws-Metal.git```


**Calibration - Intrinsics**
The first step is to find the intrinsics of your camera.
1. Print a checkerboard. You can find these from OpenCV, and print onto an A4 paper.
2. Mount the checkboard on something with a hard back, such that it may not bend.
3. Use python/robot_commander/image_processing/run_capture_images.py to capture images of the checkerboard from different positions of the camera frame, and with different tilts. Go for about 15-20 images.
4. Run python/robot_commander/image_processing/calibrate_intrinsics.py to compute intrinsics. The reprojection error should ideally be under 1 pixel if everything works as expected.

**Calibration - depth**
Get two april tags, and put them in the scene. One should lay flat against the floor. And the other at a position with lower depth.
