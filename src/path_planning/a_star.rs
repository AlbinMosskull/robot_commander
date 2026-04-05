use std::collections::HashMap;
use std::cmp::Ordering;

use pyo3::prelude::*;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::occupancy_map::occupancy_map::Position2d;
use crate::core::min_heap::MinHeap;
use crate::core::geometry_types::WorldPosition2d;


#[pyfunction]
pub fn plan_path(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Option<Vec<WorldPosition2d>> {
    let start_tuple = occ_map.convert_coordinate_to_index(world_start.x, world_start.y)?;
    let goal_tuple = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y)?;
    
    let start = Position2d { x: start_tuple.0 as isize, y: start_tuple.1 as isize };
    let goal = Position2d { x: goal_tuple.0 as isize, y: goal_tuple.1 as isize };

    let positions = plan_path_indices(occ_map, start, goal, collision_margin)?;
    let mut world_indices: Vec<WorldPosition2d> = Vec::new();
    for position in positions {
        let world_coordinates = occ_map.convert_index_to_coordinate(position.x as usize, position.y as usize);
        let world_point = WorldPosition2d{x: world_coordinates.0, y: world_coordinates.1};
        world_indices.push(world_point);
    }

    Some(world_indices)
}


fn plan_path_indices(occ_map: &OccupancyMap, start: Position2d, goal: Position2d, collision_margin: f32) -> Option<Vec<Position2d>> {
    let mut came_from: HashMap<Position2d, Position2d> = HashMap::new();
    let mut best_g_scores: HashMap<Position2d, f32> = HashMap::new();
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

        let neighbors = get_neighbors(&occ_map, current, collision_margin);
        let tentative_g_cost = best_g_scores.get(&current).expect("Should always have a value") + 1.0;  // 1.0 being cost per edge
        for neighbor in neighbors {
            if tentative_g_cost < best_g_scores.get(&neighbor).copied().unwrap_or(f32::INFINITY) {
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
    g_cost: f32,
    h_cost: f32,
}


impl Position2dWithCost {
    fn get_total_cost(&self) -> f32 {
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

fn get_neighbors(occ_map: &OccupancyMap, position: Position2d, collision_margin: f32) -> Vec<Position2d> {
    let mut neighbors = Vec::new();
    const ALLOWED_DIRECTIONS: [(isize, isize); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
    
    for (dx, dy) in ALLOWED_DIRECTIONS {
        let neighbor = Position2d{
            x: position.x + dx,
            y: position.y + dy,
        };

        if occ_map.is_valid_index(neighbor.x, neighbor.y, collision_margin) {
            println!("Adding neighbor with coords {}, {}", neighbor.x, neighbor.y);
            neighbors.push(neighbor);                                
        }           
    }

    neighbors
}


fn heuristic_cost_to_go(current: Position2d, goal: Position2d) -> f32 {
    ((goal.x - current.x).abs() + (goal.y - current.y).abs()) as f32
}





#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test_1(){
        let mut occ_map = OccupancyMap::new(3, 3, 0.1, 0.0, 0.0);
        occ_map.set_all_unoccupied();
        let start_position = Position2d {
            x: 0,
            y: 0
        };
        let goal_position = Position2d {
            x: 2,
            y: 2
        };

        let path = plan_path_indices(&occ_map, start_position, goal_position, 0.0).expect("Should find a path here");
        assert_eq!(path.len(), 5);
    }
}