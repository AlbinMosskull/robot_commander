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


def test_ray_update_gaussian_marks_hit_region_occupied():
    m = OccupancyMap(width=20, height=20, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.ray_update_gaussian(0.05, 0.05, 0.05, 0.65, 0.05)
    hit_value = m.get_cell_value(0.05, 0.65)
    assert hit_value is not None and hit_value > 0.5


def test_ray_update_gaussian_marks_cells_before_hit_as_free():
    m = OccupancyMap(width=20, height=20, resolution=0.1, origin_x=0.0, origin_y=0.0)
    m.ray_update_gaussian(0.05, 0.05, 0.05, 0.65, 0.05)
    free_value = m.get_cell_value(0.05, 0.15)
    assert free_value is not None and free_value < 0.5
