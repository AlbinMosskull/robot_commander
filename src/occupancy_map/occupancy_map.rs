use line_drawing::bresenham;

use pyo3::prelude::*;

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

    pub fn convert_coordinate_to_index(&self, world_x: f32, world_y: f32) -> Option<(usize, usize)> {
        let local_x = world_x - self.origin_x;
        let local_y = world_y - self.origin_y;

        if local_x < 0.0 || local_y < 0.0 { return None; }
        
        let x_idx = (local_x / self.resolution) as usize;
        let y_idx = (local_y / self.resolution) as usize;

        if !(self.is_within_bounds(x_idx, y_idx)) { return None; }

        Some((x_idx, y_idx))

    }
   
    pub fn ray_update(&mut self, start_world_x: f32, start_world_y: f32, end_world_x: f32, end_world_y: f32) {
        let start_grid_pos = self.convert_coordinate_to_index(start_world_x, start_world_y).expect("Start position must be within bounds");
        let end_grid_pos = self.convert_and_clip_coordinate_to_index(end_world_x, end_world_y);

        // bresenham returns an iterator that yields every (x, y) coordinate on the line.
        let ray = bresenham(                                                    
            (start_grid_pos.0 as isize, start_grid_pos.1 as isize),                  
            (end_grid_pos.0 as isize, end_grid_pos.1 as isize),    
        );

        for (x_idx, y_idx) in ray {                                                  
            let (x, y) = (x_idx as usize, y_idx as usize);
            let did_collide = (x, y) == end_grid_pos;                                
            self.update_cell(x, y, did_collide);     
        }     
    }

    pub fn is_valid_coordinate(&self, x: f32, y: f32) -> bool {
        self.convert_coordinate_to_index(x, y)
            .map(|(x_idx, y_idx)| self.is_valid_index(x_idx, y_idx))
            .unwrap_or(false)
    }

    pub fn is_valid_index(&self, x_idx: usize, y_idx: usize) -> bool {
        self.is_within_bounds(x_idx, y_idx) && self.occupancy_prob_map[y_idx][x_idx] < DEFAULT_UNCERTAINTY
    }

    pub fn set_all_unoccupied(&mut self) {
        self.occupancy_prob_map = vec![vec![0.0; self.width]; self.height]
    }
}

impl OccupancyMap {
    fn convert_and_clip_coordinate_to_index(&self, world_x: f32, world_y: f32) -> (usize, usize) {
        let local_x = (world_x - self.origin_x).max(0.0);
        let local_y = (world_y - self.origin_y).max(0.0);

        let x_idx = ((local_x / self.resolution) as usize).min(self.width);
        let y_idx = ((local_y / self.resolution) as usize).min(self.height);

        (x_idx, y_idx)
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
    fn occupancy_map_basic_1() {
        let occ_map = OccupancyMap::new(10, 10, 0.1, 0.0, 0.0);
    }

}
