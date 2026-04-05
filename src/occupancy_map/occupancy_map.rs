use line_drawing::bresenham;

use pyo3::prelude::*;

use crate::core::geometry_types::WorldPosition2d;


// Position within the OccupancyMap grid
#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct Position2d {
    pub x: isize,
    pub y: isize,
}

const DEFAULT_UNCERTAINTY: f32 = 0.5;
const UPDATE_IF_COLLISION: f32 = 0.85;
const UPDATE_IF_FREE: f32 = -0.5;

#[pyclass]
pub struct OccupancyMap {
    width: usize,     
    height: usize,
    resolution: f32,  // m per box side
    // The world-space coordinate of the [0,0] cell
    origin_x: f32,
    origin_y: f32,
    occupancy_prob_map: Vec<Vec<f32>>,
}

#[pymethods]
impl OccupancyMap {
    #[new]
    pub fn new(width: usize, height: usize, resolution: f32, origin_x: f32, origin_y: f32) -> OccupancyMap {
        OccupancyMap{
            resolution,
            width,
            height,
            origin_x,
            origin_y,
            occupancy_prob_map: vec![vec![DEFAULT_UNCERTAINTY; width]; height],
        }
    }
   
    pub fn ray_update(&mut self, start_world_x: f32, start_world_y: f32, end_world_x: f32, end_world_y: f32, did_collide: bool) {
        let start_grid_pos = self.convert_coordinate_to_index(start_world_x, start_world_y).expect("Start position must be within bounds");
        let end_grid_pos = self.convert_and_clip_coordinate_to_index(end_world_x, end_world_y);

        let ray = bresenham(
            (start_grid_pos.x, start_grid_pos.y),
            (end_grid_pos.x, end_grid_pos.y),
        );

        for (x_idx, y_idx) in ray {
            let is_collision = (x_idx, y_idx) == (end_grid_pos.x, end_grid_pos.y) && did_collide;
            self.update_cell(x_idx as usize, y_idx as usize, is_collision);
        }
    }

    pub fn is_valid_coordinate(&self, x: f32, y: f32, collision_margin: f32) -> bool {
        self.convert_coordinate_to_index(x, y)
            .map(|position| self.is_valid_index(position, collision_margin))
            .unwrap_or(false)
    }

    pub fn set_all_unoccupied(&mut self) {
        self.occupancy_prob_map = vec![vec![0.0; self.width]; self.height]
    }

    pub fn get_grid(&self) -> Vec<Vec<f32>> {
        self.occupancy_prob_map.clone()
    }
}

impl OccupancyMap {
    pub fn convert_coordinate_to_index(&self, world_x: f32, world_y: f32) -> Option<Position2d> {
        let local_x = world_x - self.origin_x;
        let local_y = world_y - self.origin_y;

        if local_x < 0.0 || local_y < 0.0 { return None; }

        let x_idx = (local_x / self.resolution) as usize;
        let y_idx = (local_y / self.resolution) as usize;

        if !(self.is_within_bounds(x_idx, y_idx)) { return None; }

        Some(Position2d { x: x_idx as isize, y: y_idx as isize })
    }

    pub fn convert_index_to_coordinate(&self, position: Position2d) -> WorldPosition2d {
        WorldPosition2d {
            x: position.x as f32 * self.resolution + self.origin_x,
            y: position.y as f32 * self.resolution + self.origin_y,
        }
    }

    pub fn is_valid_index(&self, position: Position2d, collision_margin: f32) -> bool {
        if position.x < 0 || position.y < 0 { return false; }
        let x_idx_to_check = position.x as usize;
        let y_idx_to_check = position.y as usize;

        let cells_to_check = (collision_margin / self.resolution) as usize + 1;

        for x_idx in x_idx_to_check.saturating_sub(cells_to_check)..x_idx_to_check+cells_to_check {
            for y_idx in y_idx_to_check.saturating_sub(cells_to_check)..y_idx_to_check+cells_to_check {
                let is_valid = self.is_within_bounds(x_idx, y_idx) && self.occupancy_prob_map[y_idx][x_idx] < DEFAULT_UNCERTAINTY;
                if !is_valid { return false; }
            }
        }

        true
    }

    fn convert_and_clip_coordinate_to_index(&self, world_x: f32, world_y: f32) -> Position2d {
        let local_x = (world_x - self.origin_x).max(0.0);
        let local_y = (world_y - self.origin_y).max(0.0);

        Position2d {
            x: ((local_x / self.resolution) as usize).min(self.width) as isize,
            y: ((local_y / self.resolution) as usize).min(self.height) as isize,
        }
    }
    
    fn update_cell(&mut self, x_idx: usize, y_idx: usize, did_collide: bool) {
        let prob_update = if did_collide {UPDATE_IF_COLLISION} else {UPDATE_IF_FREE};
        self.occupancy_prob_map[y_idx][x_idx] += prob_update;
    }

    fn is_within_bounds(&self, x_idx: usize, y_idx: usize) -> bool {
        x_idx < self.width && y_idx < self.height
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert_coordinate_to_index_maps_correctly() {
        let occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        assert_eq!(occ_map.convert_coordinate_to_index(0.05, 0.15), Some(Position2d { x: 0, y: 1 }));
    }

    #[test]
    fn convert_coordinate_to_index_returns_none_outside_bounds() {
        let occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        assert_eq!(occ_map.convert_coordinate_to_index(-0.1, 0.0), None);
        assert_eq!(occ_map.convert_coordinate_to_index(0.0, 1.1), None);
    }

    #[test]
    fn convert_coordinate_to_index_respects_origin() {
        let occ_map = OccupancyMap::new(10, 10, 0.1, 1.0, 2.0);
        assert_eq!(occ_map.convert_coordinate_to_index(1.00, 2.00), Some(Position2d { x: 0, y: 0 }));
        assert_eq!(occ_map.convert_coordinate_to_index(1.05, 2.05), Some(Position2d { x: 0, y: 0 }));
        assert_eq!(occ_map.convert_coordinate_to_index(0.9, 2.0), None);
    }

    #[test]
    fn ray_update_marks_collision_cell_as_occupied() {
        let mut occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        occ_map.ray_update(0.05, 0.05, 0.05, 0.35, true);
        let grid = occ_map.get_grid();
        assert!(grid[3][0] > DEFAULT_UNCERTAINTY, "Collision cell should be above default uncertainty");
        assert!(grid[0][0] < DEFAULT_UNCERTAINTY, "Free cells along ray should be below default uncertainty");
    }

    #[test]
    fn is_valid_coordinate_false_after_collision() {
        let mut occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        occ_map.ray_update(0.05, 0.05, 0.05, 0.35, true);
        assert!(!occ_map.is_valid_coordinate(0.05, 0.35, 0.0));
        assert!(occ_map.is_valid_coordinate(0.05, 0.05, 0.0));
    }
}
