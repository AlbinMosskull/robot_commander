from robot_commander import OccupancyMap


def test_is_valid_coordinate_unknown_by_default():
    m = OccupancyMap(width=10, height=10, resolution=0.1, origin_x=0.0, origin_y=0.0)
    assert not m.is_valid_coordinate(0.05, 0.05, 0.0)


def test_set_all_unoccupied_makes_cells_valid():
    m = OccupancyMap(width=10, height=10, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.set_all_unoccupied()
    assert m.is_valid_coordinate(0.05, 0.05, 0.0)


def test_ray_update_marks_endpoint_occupied():
    m = OccupancyMap(width=10, height=10, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.set_all_unoccupied()
    m.ray_update(0.05, 0.05, 0.95, 0.05, True)
    assert not m.is_valid_coordinate(0.95, 0.05, 0.0)


def test_ray_update_marks_free_cells_valid():
    m = OccupancyMap(width=10, height=10, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.set_all_unoccupied()
    m.ray_update(0.05, 0.05, 0.95, 0.05, False)
    assert m.is_valid_coordinate(0.05, 0.05, 0.0)
