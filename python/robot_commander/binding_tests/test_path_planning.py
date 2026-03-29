from robot_commander import OccupancyMap, WorldPosition2d, plan_path


def test_world_position2d_construct():
    pos = WorldPosition2d(1.0, 2.0)
    assert pos.x == 1.0
    assert pos.y == 2.0


def test_plan_path_finds_path():
    m = OccupancyMap(width=10, height=10, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.set_all_unoccupied()
    start = WorldPosition2d(0.05, 0.05)
    goal = WorldPosition2d(0.25, 0.05)
    path = plan_path(m, start, goal, 0.1)
    assert path is not None
    assert len(path) > 0
