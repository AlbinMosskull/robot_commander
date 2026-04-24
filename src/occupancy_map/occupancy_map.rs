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
const MAX_CELL_VALUE: f32 = 5.0;

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

    pub fn ray_update_gaussian(
        &mut self,
        start_world_x: f32,
        start_world_y: f32,
        end_world_x: f32,
        end_world_y: f32,
        sigma_m: f32,
    ) {
        let hit_dist = ((end_world_x - start_world_x).powi(2) + (end_world_y - start_world_y).powi(2)).sqrt();
        if hit_dist == 0.0 {
            return;
        }

        let dir_x = (end_world_x - start_world_x) / hit_dist;
        let dir_y = (end_world_y - start_world_y) / hit_dist;
        let tail = 3.0 * sigma_m;

        let start_grid = match self.convert_coordinate_to_index(start_world_x, start_world_y) {
            Some(p) => p,
            None => return,
        };
        let ext_end_grid = self.convert_and_clip_coordinate_to_index(
            end_world_x + dir_x * tail,
            end_world_y + dir_y * tail,
        );

        for (x_idx, y_idx) in bresenham((start_grid.x, start_grid.y), (ext_end_grid.x, ext_end_grid.y)) {
            let cell_world_x = x_idx as f32 * self.resolution + self.origin_x + self.resolution * 0.5;
            let cell_world_y = y_idx as f32 * self.resolution + self.origin_y + self.resolution * 0.5;
            let dist_along_ray = (cell_world_x - start_world_x) * dir_x + (cell_world_y - start_world_y) * dir_y;

            let delta = if dist_along_ray < hit_dist - tail {
                UPDATE_IF_FREE
            } else if dist_along_ray <= hit_dist + tail {
                let diff = dist_along_ray - hit_dist;
                let weight = (-0.5 * (diff / sigma_m).powi(2)).exp();
                weight * UPDATE_IF_COLLISION
            } else {
                break;
            };

            let x = x_idx as usize;
            let y = y_idx as usize;
            if x < self.width && y < self.height {
                let new_value = self.occupancy_prob_map[y][x] + delta;
                self.occupancy_prob_map[y][x] = new_value.clamp(0.0, MAX_CELL_VALUE);
            }
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

    pub fn set_grid(&mut self, grid: Vec<Vec<f32>>) {
        self.occupancy_prob_map = grid;
    }

    pub fn get_cell_value(&self, world_x: f32, world_y: f32) -> Option<f32> {
        let pos = self.convert_coordinate_to_index(world_x, world_y)?;
        Some(self.occupancy_prob_map[pos.y as usize][pos.x as usize])
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
        if !self.is_within_bounds(x_idx_to_check, y_idx_to_check) { return false; }

        let cells_to_check = if collision_margin > 0.0 {
            (collision_margin / self.resolution) as usize + 1                                              
        } else {
            0  // Zero margin should entail only one cell checked.                                                                                    
        };

        for x_idx in x_idx_to_check.saturating_sub(cells_to_check)..x_idx_to_check+cells_to_check+1 {
            for y_idx in y_idx_to_check.saturating_sub(cells_to_check)..y_idx_to_check+cells_to_check+1 {
                let dx = x_idx as isize - x_idx_to_check as isize;
                let dy = y_idx as isize - y_idx_to_check as isize;
                if dx * dx + dy * dy > (cells_to_check * cells_to_check) as isize { continue; }
                if !self.is_within_bounds(x_idx, y_idx) { continue; }
                if self.occupancy_prob_map[y_idx][x_idx] >= DEFAULT_UNCERTAINTY { return false; }
            }
        }

        true
    }

    fn convert_and_clip_coordinate_to_index(&self, world_x: f32, world_y: f32) -> Position2d {
        let local_x = (world_x - self.origin_x).max(0.0);
        let local_y = (world_y - self.origin_y).max(0.0);

        Position2d {
            x: ((local_x / self.resolution) as usize).min(self.width-1) as isize,
            y: ((local_y / self.resolution) as usize).min(self.height-1) as isize,
        }
    }
    
    pub(crate) fn update_cell(&mut self, x_idx: usize, y_idx: usize, did_collide: bool) {
        let prob_update = if did_collide { UPDATE_IF_COLLISION } else { UPDATE_IF_FREE };
        let new_value = self.occupancy_prob_map[y_idx][x_idx] + prob_update;
        self.occupancy_prob_map[y_idx][x_idx] = new_value.clamp(0.0, MAX_CELL_VALUE);
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
    fn is_valid_index_detects_collision_exactly_on_positive_side_of_margin() {
        let mut occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        occ_map.set_all_unoccupied();
        occ_map.update_cell(2, 0, true);
        // collision_margin=0.1, resolution=0.1, should check 2 cells, so (2,0) should be within margin
        let border_resolution = 0.1;
        let under_resolution = 0.1 - 1e-3;
        assert!(occ_map.is_valid_coordinate(0.05, 0.05, under_resolution));
        assert!(!occ_map.is_valid_coordinate(0.05, 0.05, border_resolution));
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
    fn is_valid_index_margin_is_circular_not_square() {
        let mut occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        occ_map.set_all_unoccupied();
        // Index (2,2) is distance sqrt(8) ~= 2.83 from (0,0), which is outside a circle of radius 2 but inside the square
        occ_map.update_cell(2, 2, true);
        assert!(occ_map.is_valid_coordinate(0.05, 0.05, 0.1));
    }

    #[test]
    fn is_valid_coordinate_false_after_collision() {
        let mut occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
        occ_map.ray_update(0.05, 0.05, 0.05, 0.35, true);
        assert!(!occ_map.is_valid_coordinate(0.05, 0.35, 0.0));
        assert!(occ_map.is_valid_coordinate(0.05, 0.05, 0.0));
    }

    #[test]
    fn ray_update_gaussian_raises_hit_cell_above_default() {
        let mut occ_map = OccupancyMap::new(20, 20, 0.1, 0.0, 0.0);
        occ_map.ray_update_gaussian(0.05, 0.05, 0.05, 0.65, 0.05);
        let grid = occ_map.get_grid();
        assert!(grid[6][0] > DEFAULT_UNCERTAINTY, "Hit cell should be above default uncertainty");
    }

    #[test]
    fn ray_update_gaussian_marks_free_cells_before_hit() {
        let mut occ_map = OccupancyMap::new(20, 20, 0.1, 0.0, 0.0);
        occ_map.ray_update_gaussian(0.05, 0.05, 0.05, 0.65, 0.05);
        let grid = occ_map.get_grid();
        assert!(grid[1][0] < DEFAULT_UNCERTAINTY, "Cell well before hit should be marked free");
    }

    #[test]
    fn ray_update_gaussian_hit_cell_higher_than_shoulder() {
        let mut occ_map = OccupancyMap::new(20, 20, 0.1, 0.0, 0.0);
        occ_map.ray_update_gaussian(0.05, 0.05, 0.05, 0.65, 0.05);
        let grid = occ_map.get_grid();
        // Cell at exact hit (row 6) should be higher than cells at shoulders (rows 4, 8)
        assert!(grid[6][0] > grid[4][0]);
        assert!(grid[6][0] > grid[8][0]);
    }
}
