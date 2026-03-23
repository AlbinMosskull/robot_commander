pub struct OccupancyMap {
    num_rows: usize,
    num_cols: usize,
    is_occupied: Vec<Vec<bool>>,
}

impl OccupancyMap {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        OccupancyMap{
            num_rows:num_rows,
            num_cols:num_cols,
            is_occupied: vec![vec![false; num_cols]; num_rows]
        }
    }
    fn is_within_bounds(&self, x: usize, y: usize) -> bool {
        let is_x_valid = x >= 0 && x < self.num_cols;
        let is_y_valid = y >= 0 && y < self.num_rows;
        is_x_valid && is_y_valid
    }

    pub fn is_valid(&self, x: usize, y: usize) -> bool {
        self.is_within_bounds(x, y) && !self.is_occupied[y][x]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn occupancy_map_1_element() {
        let occ_map = OccupancyMap::new(1, 1);
        assert!(!occ_map.is_occupied[0][0]);
    }

    #[test]
    fn occupancy_map_mutate() {
        let mut occ_map = OccupancyMap::new(1, 1);
        assert!(!occ_map.is_occupied[0][0]);
        occ_map.is_occupied[0][0] = true;
        assert!(occ_map.is_occupied[0][0]);
    }
}
