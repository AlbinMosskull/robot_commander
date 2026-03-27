"""
Interactive depth evaluation script

1. First a camera should be initialized and warmed up
2. Then, a frame should be captured and the user should be allowed to draw a bounding box on it
3. Depth should then be captured within this region. (initially using depth anything)
4. It should then be converted to a point cloud
5. Then RANSAC should produce a plane fit to this point cloud
6. Finally, the length and the width of the plane should be printed, and std and max outlier from the plane.

"""