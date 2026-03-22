use std::collections::HashMap;
use std::cmp::Ordering;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::core::min_heap::MinHeap;


// TODO move to a common data types folder.
#[derive(Copy, Clone)] 
pub struct Position {
    x: usize,
    y: usize,
}

#[derive(Copy, Clone)] 
struct PositionWithCost {
    position: Position,
    from_start_cost: f64,  // g-cost
    heuristic_cost_to_go: f64,  // h-cost
}

impl PositionWithCost {
    pub fn get_total_cost(&self) -> f64 {
        self.from_start_cost + self.heuristic_cost_to_go
    }
}

impl Ord for PositionWithCost {
    fn cmp(&self, other: &Self) -> Ordering {
        let diff = self.get_total_cost() - other.get_total_cost();
        if diff < 0.0 {
            return Ordering::Less;
        } else if diff > 0.0 {
            return Ordering::Greater
        } else {
            return Ordering::Equal
        }
    }
}

impl PartialOrd for PositionWithCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for PositionWithCost {}

impl PartialEq for PositionWithCost {
    fn eq(&self, other: &Self) -> bool {
        const EPS: f64 = 1e-6;
        (self.get_total_cost() - other.get_total_cost()).abs() < EPS
    }
}


pub fn plan_path(occ_map: OccupancyMap, start: Position, goal: Position) -> Vec<Position> {
    let mut came_from: HashMap<Position, Position> = HashMap::new();
    let mut open_set: MinHeap<PositionWithCost> = MinHeap::new();
    
    
    Vec::new()
}

fn heuristic_cost_to_go(current: Position, goal: Position) -> usize {
    (goal.x - current.x)^2 + (goal.y - current.y)^2
}


fn hello_world() {
    let mut occ_map = OccupancyMap::new(3, 3);
    let mut priority_queue: MinHeap<i32> = MinHeap::new();
    priority_queue.insert(1);

    println!("Hello, world!");

}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dummy() {
        hello_world();
        assert!(true);
    }
}