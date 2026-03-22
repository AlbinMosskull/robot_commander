use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::Ordering;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::core::min_heap::MinHeap;


// TODO move to a common data types folder.
#[derive(Copy, Clone, Hash)] 
pub struct Position {
    x: usize,
    y: usize,
}

impl Eq for Position {}

impl PartialEq for Position {
    fn eq(&self, other: &Self) -> bool {
        self.x == other.x && self.y == other.y
    }
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
            return Ordering::Greater;
        } else {
            return Ordering::Equal;
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

fn reconstruct_path(came_from: &HashMap<Position, Position>, final_position: Position) -> Vec<Position> {
    let full_path: Vec<Position> = Vec::new();

    let mut current_position = final_position;
    // loop {
    //     match came_from[current_position]


    // }
    full_path
}

fn get_neighbors(occ_map: &OccupancyMap, position: Position) -> Vec<PositionWithCost> {
    let neighbors: Vec<PositionWithCost> = Vec::new();
    neighbors
}


pub fn plan_path(occ_map: OccupancyMap, start: Position, goal: Position) -> Vec<Position> {
    let mut closed: HashSet<Position> = HashSet::new();
    let mut came_from: HashMap<Position, Position> = HashMap::new();
    let mut open_set: MinHeap<PositionWithCost> = MinHeap::new();

    let costed_start = PositionWithCost {
        position: start,
        from_start_cost: 0.0,
        heuristic_cost_to_go: heuristic_cost_to_go(start, goal),
    };
    open_set.insert(costed_start);
    
    while open_set.len() != 0 {
        let current = open_set.extract_min().expect("While loop checks for empty");

        if current.position == goal {
            return reconstruct_path(&came_from, current.position);
        }

        let mut neighbors = get_neighbors(&occ_map, current.position);
        for mut neighbor in neighbors {
            let tenative_cost = current.from_start_cost + 1.0;  // 1 being cost per edge
            if tenative_cost < neighbor.from_start_cost {
                // came_from[&neighbor.position] = current.position;
                if let Some(val) = came_from.get_mut(&neighbor.position) { *val = current.position; }; // No clue what is happening here
                neighbor.from_start_cost = tenative_cost;
                // if neighbor ! in open_set {  // TODO need to support this
                //     open_set.insert(neighbor);
                // }
            }
        }
    }

    Vec::new()  // failure!
}

fn heuristic_cost_to_go(current: Position, goal: Position) -> f64 {
    ((goal.x - current.x).pow(2) + (goal.y - current.y).pow(2)) as f64
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