use std::collections::HashMap;
use std::cmp::Ordering;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::core::min_heap::MinHeap;


#[derive(Copy, Clone, Hash, PartialEq, Eq)] 
pub struct Position2d {
    pub x: isize,
    pub y: isize,
}


pub fn plan_path(occ_map: &OccupancyMap, start: Position2d, goal: Position2d) -> Option<Vec<Position2d>> {
    let mut came_from: HashMap<Position2d, Position2d> = HashMap::new();
    let mut best_g_scores: HashMap<Position2d, f64> = HashMap::new();
    let mut open_set: MinHeap<Position2dWithCost> = MinHeap::new();

    let costed_start = Position2dWithCost {
        position: start,
        g_cost: 0.0,
        h_cost: heuristic_cost_to_go(start, goal),
    };
    open_set.insert(costed_start);
    best_g_scores.insert(start, 0.0);
    
    while open_set.len() > 0 {
        let costed_current = open_set.extract_min().expect("While loop checks for empty");
        let current = costed_current.position;

        println!("Investigating position {} {}", current.x, current.y);

        if costed_current.g_cost > best_g_scores.get(&current).copied().expect("Should always have a value") {
            continue;  // A better path to the node has already been added, so we do not need to explore this one.
        }

        if current == goal {
            return Some(reconstruct_path(&came_from, current));
        }

        let neighbors = get_neighbors(&occ_map, current);
        let tentative_g_cost = best_g_scores.get(&current).expect("Should always have a value") + 1.0;  // 1.0 being cost per edge
        for neighbor in neighbors {
            if tentative_g_cost < best_g_scores.get(&neighbor).copied().unwrap_or(f64::INFINITY) {
                came_from.insert(neighbor, current);  // insert will update if key is present.
                let neighbor_with_cost = Position2dWithCost{
                    position: neighbor,
                    g_cost: tentative_g_cost,
                    h_cost: heuristic_cost_to_go(neighbor, goal)
                };
                
                best_g_scores.insert(neighbor, tentative_g_cost);
                open_set.insert(neighbor_with_cost);
            }
        }
    }

    // Could not reach the goal
    None
}



#[derive(Copy, Clone)] 
struct Position2dWithCost {
    position: Position2d,
    g_cost: f64,
    h_cost: f64,
}


impl Position2dWithCost {
    fn get_total_cost(&self) -> f64 {
        self.g_cost + self.h_cost
    }
}

impl Ord for Position2dWithCost {
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

impl PartialOrd for Position2dWithCost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for Position2dWithCost {}

impl PartialEq for Position2dWithCost {
    fn eq(&self, other: &Self) -> bool {
        self.get_total_cost() == other.get_total_cost()
    }
}

fn reconstruct_path(came_from: &HashMap<Position2d, Position2d>, final_position: Position2d) -> Vec<Position2d> {
    let mut full_path: Vec<Position2d> = Vec::new();

    let mut current_position = final_position;
    loop {
        full_path.push(current_position);

        let next = came_from.get(&current_position);

        match next {
            None => break,
            Some(next) => current_position = *next,
        }
    }
    full_path.reverse();
    full_path
}

fn get_neighbors(occ_map: &OccupancyMap, position: Position2d) -> Vec<Position2d> {
    let mut neighbors = Vec::new();
    const ALLOWED_DIRECTIONS: [(isize, isize); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
    
    for (dx, dy) in ALLOWED_DIRECTIONS {
        let neighbor = Position2d{
            x: position.x + dx,
            y: position.y + dy,
        };

        if let (Ok(x), Ok(y)) = (neighbor.x.try_into(),neighbor.y.try_into()) {                                         
            if occ_map.is_valid(x, y) {
                println!("Adding neighbor with coords {}, {}", neighbor.x, neighbor.y);
                neighbors.push(neighbor);                                
            }           
        }
    }

    neighbors
}


fn heuristic_cost_to_go(current: Position2d, goal: Position2d) -> f64 {
    ((goal.x - current.x).abs() + (goal.y - current.y).abs()) as f64
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test_1(){
        let occ_map = OccupancyMap::new(3, 3);
        let start_position = Position2d {
            x: 0,
            y: 0
        };
        let goal_position = Position2d {
            x: 2,
            y: 2
        };

        let path = plan_path(&occ_map, start_position, goal_position).expect("Should find a path here");
        assert_eq!(path.len(), 5);
    }
}