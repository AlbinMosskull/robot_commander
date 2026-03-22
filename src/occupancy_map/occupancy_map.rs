pub struct OccupancyMap {
    is_occupied: Vec<Vec<bool>>,
}

impl OccupancyMap {
    pub fn new(num_rows: usize, num_cols: usize) -> Self {
        OccupancyMap{
            is_occupied: vec![vec![false; num_cols]; num_rows]
        }
    }
    pub fn is_position_occupied(&self, x: usize, y: usize) -> bool {
        self.is_occupied[y][x]
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
