use std::collections::HashMap;
use std::cmp::Ordering;

use pyo3::prelude::*;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::occupancy_map::occupancy_map::Position2d;
use crate::core::min_heap::MinHeap;
use crate::core::geometry_types::WorldPosition2d;


#[pyfunction]
pub fn plan_path(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Option<Vec<WorldPosition2d>> {
    let start = occ_map.convert_coordinate_to_index(world_start.x, world_start.y)?;
    let goal = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y)?;

    let positions = plan_path_indices(occ_map, start, goal, collision_margin)?;
    Some(positions.into_iter().map(|p| occ_map.convert_index_to_coordinate(p)).collect())
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

        if occ_map.is_valid_index(neighbor, collision_margin) {
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

    fn make_map(width: usize, height: usize) -> OccupancyMap {
        let mut occ_map = OccupancyMap::new(width, height, 0.1, 0.0, 0.0);
        occ_map.set_all_unoccupied();
        occ_map
    }

    #[test]
    fn finds_path_in_open_space() {
        let occ_map = make_map(3, 3);
        let path = plan_path_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0).expect("Should find a path");
        assert_eq!(path.len(), 5);
    }

    #[test]
    fn start_equals_goal_returns_single_node_path() {
        let occ_map = make_map(5, 5);
        let path = plan_path_indices(&occ_map, Position2d { x: 2, y: 2 }, Position2d { x: 2, y: 2 }, 0.0).expect("Should find a path");
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn path_routes_around_obstacle() {
        let mut occ_map = make_map(5, 3);
        // Wall at x=2 blocking rows y=0 and y=1, leaving y=2 as the only passage
        occ_map.update_cell(2, 0, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.0).expect("Should route around obstacle");
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn returns_none_when_goal_is_unreachable() {
        let mut occ_map = make_map(3, 3);
        // Block all approaches to (2,2)
        occ_map.update_cell(1, 2, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_path_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0);
        assert!(path.is_none());
    }

    #[test]
    fn collision_margin_blocks_path_through_narrow_gap() {
        let mut occ_map = make_map(5, 3);
        // One-cell gap at (2,1) between obstacles at (2,0) and (2,2)
        occ_map.update_cell(2, 0, true);
        occ_map.update_cell(2, 2, true);
        assert!(plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.0).is_some());
        assert!(plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.05).is_none());
    }
}