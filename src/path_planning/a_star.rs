use std::collections::HashMap;
use std::collections::HashSet;
use std::cmp::Ordering;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::core::min_heap::MinHeap;


// TODO move to a common data types folder.
#[derive(Copy, Clone, Hash)] 
pub struct Position {
    x: isize,
    y: isize,
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
    let mut full_path: Vec<Position> = Vec::new();

    let mut current_position = final_position;
    loop {
        full_path.push(current_position);

        let next = came_from.get(&current_position);

        match next {
            None => break,
            Some(next) => current_position = *next,
        }
    }
    full_path
}

fn get_neighbors(occ_map: &OccupancyMap, position: Position) -> Vec<Position> {
    let mut neighbors: Vec<Position> = Vec::new();
    let allowed_directions = [(0, 1), (1, 0), (0, -1), (-1, 0)];
    
    for (dx, dy) in allowed_directions {
        let neighbor = Position{
            x: position.x + dx,
            y: position.y + dy,
        };

        if neighbor.x < 0 || neighbor.y < 0 {
            continue;  // usize is crashing out below due to negative values
        }

        let neighbor_is_valid = occ_map.is_valid(neighbor.x.try_into().unwrap(), neighbor.y.try_into().unwrap());

        println!("Neighbor with coords {}, {}, is_valid {}", neighbor.x, neighbor.y, neighbor_is_valid);

        if neighbor_is_valid {
            neighbors.push(neighbor)
        }
    }

    neighbors
}


pub fn plan_path(occ_map: OccupancyMap, start: Position, goal: Position) -> Vec<Position> {
    let mut came_from: HashMap<Position, Position> = HashMap::new();
    let mut best_g_scores: HashMap<Position, f64> = HashMap::new();
    let mut open_set: MinHeap<PositionWithCost> = MinHeap::new();

    let costed_start = PositionWithCost {
        position: start,
        from_start_cost: 0.0,
        heuristic_cost_to_go: heuristic_cost_to_go(start, goal),
    };
    open_set.insert(costed_start);
    best_g_scores.insert(start, 0.0);
    
    while open_set.len() != 0 {
        let costed_current = open_set.extract_min().expect("While loop checks for empty");
        let current = costed_current.position;

        println!("Investigating position {} {}", current.x, current.y);

        if costed_current.from_start_cost > *best_g_scores.get(&current).expect("Should always have a value") {
            continue;  // Stale node.
        }

        if current == goal {
            return reconstruct_path(&came_from, current);
        }

        let neighbors = get_neighbors(&occ_map, current);
        for neighbor in neighbors {
            let tenative_g_cost = best_g_scores.get(&current).expect("Should always have a value") + 1.0;  // 1 being cost per edge
            if tenative_g_cost < *best_g_scores.get(&neighbor).unwrap_or(&f64::INFINITY) {
                came_from.insert(neighbor, current);  // Will update if not present.
                let neighbor_with_cost = PositionWithCost{
                    position: neighbor,
                    from_start_cost: tenative_g_cost,
                    heuristic_cost_to_go: heuristic_cost_to_go(current, goal)
                };
                
                best_g_scores.insert(neighbor, tenative_g_cost);
                open_set.insert(neighbor_with_cost);
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

    let start_position = Position {
        x: 0,
        y: 0
    };
    let goal_position = Position {
        x: 2,
        y: 2
    };

    let path = plan_path(occ_map, start_position, goal_position);
    println!("Length of path {}", path.len());


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

    #[test]
    fn dummy_2(){
        let mut occ_map = OccupancyMap::new(3, 3);
        let mut priority_queue: MinHeap<i32> = MinHeap::new();
        priority_queue.insert(1);

        let start_position = Position {
            x: 0,
            y: 0
        };
        let goal_position = Position {
            x: 2,
            y: 2
        };

        let path = plan_path(occ_map, start_position, goal_position);
        assert_eq!(path.len(), 5);
    }
}