use std::collections::HashMap;
use std::cmp::Ordering;

use crate::occupancy_map::occupancy_map::Position2d;


#[derive(Copy, Clone)]
pub struct Position2dWithCost {
    pub position: Position2d,
    pub g_cost: f32,
    pub h_cost: f32,
}

impl Position2dWithCost {
    pub fn total_cost(&self) -> f32 {
        self.g_cost + self.h_cost
    }
}

impl Ord for Position2dWithCost {
    fn cmp(&self, other: &Self) -> Ordering {
        let diff = self.total_cost() - other.total_cost();
        if diff < 0.0 { Ordering::Less } else if diff > 0.0 { Ordering::Greater } else { Ordering::Equal }
    }
}

impl PartialOrd for Position2dWithCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Position2dWithCost {}

impl PartialEq for Position2dWithCost {
    fn eq(&self, other: &Self) -> bool {
        self.total_cost() == other.total_cost()
    }
}

pub fn reconstruct_path(came_from: &HashMap<Position2d, Position2d>, final_position: Position2d) -> Vec<Position2d> {
    let mut path: Vec<Position2d> = Vec::new();
    let mut current = final_position;
    loop {
        path.push(current);
        match came_from.get(&current) {
            None => break,
            Some(next) => current = *next,
        }
    }
    path.reverse();
    path
}
