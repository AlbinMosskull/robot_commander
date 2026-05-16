use std::collections::HashMap;

use pyo3::prelude::*;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::occupancy_map::occupancy_map::Position2d;
use crate::core::min_heap::MinHeap;
use crate::core::geometry_types::WorldPosition2d;
use super::common::{Position2dWithCost, reconstruct_path};


#[pyfunction]
pub fn plan_path(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Option<Vec<WorldPosition2d>> {
    let start = occ_map.convert_coordinate_to_index(world_start.x, world_start.y)?;
    let goal = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y)?;

    let positions = plan_path_indices(occ_map, start, goal, collision_margin)?;
    Some(positions.into_iter().map(|p| occ_map.convert_index_to_coordinate(p)).collect())
}

#[pyfunction]
pub fn plan_path_towards_goal(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Vec<WorldPosition2d> {
    let start = occ_map.convert_coordinate_to_index(world_start.x, world_start.y).expect("Must give a position within world bounds");
    let goal = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y).expect("Must give a position within world bounds");

    let positions = plan_path_towards_goal_indices(occ_map, start, goal, collision_margin);
    positions.into_iter().map(|p| occ_map.convert_index_to_coordinate(p)).collect()
}

fn plan_path_indices(occ_map: &OccupancyMap, start: Position2d, goal: Position2d, collision_margin: f32) -> Option<Vec<Position2d>> {
    let result = plan_path_towards_goal_indices(occ_map, start, goal, collision_margin);
    (result.last() == Some(&goal)).then_some(result)
}

fn plan_path_towards_goal_indices(occ_map: &OccupancyMap, start: Position2d, goal: Position2d, collision_margin: f32) -> Vec<Position2d> {
    let mut came_from: HashMap<Position2d, Position2d> = HashMap::new();
    let mut best_g_scores: HashMap<Position2d, f32> = HashMap::new();
    let mut open_set: MinHeap<Position2dWithCost> = MinHeap::new();

    open_set.insert(Position2dWithCost {
        position: start,
        g_cost: 0.0,
        h_cost: manhattan_distance(start, goal),
    });
    best_g_scores.insert(start, 0.0);

    while open_set.len() > 0 {
        let costed_current = open_set.extract_min().expect("While loop checks for empty");
        let current = costed_current.position;

        if costed_current.g_cost > best_g_scores.get(&current).copied().expect("Should always have a value") {
            continue;
        }

        if current == goal {
            return reconstruct_path(&came_from, current);
        }

        let tentative_g_cost = best_g_scores.get(&current).expect("Should always have a value") + 1.0;
        for neighbor in get_neighbors(occ_map, current, collision_margin) {
            if tentative_g_cost < best_g_scores.get(&neighbor).copied().unwrap_or(f32::INFINITY) {
                came_from.insert(neighbor, current);
                best_g_scores.insert(neighbor, tentative_g_cost);
                open_set.insert(Position2dWithCost {
                    position: neighbor,
                    g_cost: tentative_g_cost,
                    h_cost: manhattan_distance(neighbor, goal),
                });
            }
        }
    }

    let closest_to_goal = *best_g_scores.keys()
        .min_by(|a, b| manhattan_distance(**a, goal).partial_cmp(&manhattan_distance(**b, goal)).unwrap())
        .expect("best_g_scores should never be empty");
    reconstruct_path(&came_from, closest_to_goal)
}

fn get_neighbors(occ_map: &OccupancyMap, position: Position2d, collision_margin: f32) -> Vec<Position2d> {
    const DIRECTIONS: [(isize, isize); 4] = [(0, 1), (1, 0), (0, -1), (-1, 0)];
    DIRECTIONS.iter()
        .map(|(dx, dy)| Position2d { x: position.x + dx, y: position.y + dy })
        .filter(|neighbor| occ_map.is_valid_index(*neighbor, collision_margin))
        .collect()
}

fn manhattan_distance(a: Position2d, b: Position2d) -> f32 {
    ((b.x - a.x).abs() + (b.y - a.y).abs()) as f32
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
        occ_map.update_cell(2, 0, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.0).expect("Should route around obstacle");
        assert_eq!(path.len(), 7);
    }

    #[test]
    fn returns_none_when_goal_is_unreachable() {
        let mut occ_map = make_map(3, 3);
        occ_map.update_cell(1, 2, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_path_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0);
        assert!(path.is_none());
    }

    #[test]
    fn towards_goal_reaches_goal_when_reachable() {
        let occ_map = make_map(5, 5);
        let path = plan_path_towards_goal_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 4, y: 4 }, 0.0);
        assert_eq!(*path.last().unwrap(), Position2d { x: 4, y: 4 });
    }

    #[test]
    fn towards_goal_ends_at_closest_reachable_node_when_goal_blocked() {
        let mut occ_map = make_map(3, 3);
        occ_map.update_cell(1, 2, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_path_towards_goal_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0);
        assert!(!path.is_empty());
        assert_ne!(*path.last().unwrap(), Position2d { x: 2, y: 2 });
        assert_eq!(manhattan_distance(*path.last().unwrap(), Position2d { x: 2, y: 2 }), 2.0);
    }

    #[test]
    fn towards_goal_returns_single_node_when_start_equals_goal() {
        let occ_map = make_map(5, 5);
        let path = plan_path_towards_goal_indices(&occ_map, Position2d { x: 2, y: 2 }, Position2d { x: 2, y: 2 }, 0.0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], Position2d { x: 2, y: 2 });
    }

    #[test]
    fn collision_margin_blocks_path_through_narrow_gap() {
        let mut occ_map = make_map(5, 3);
        occ_map.update_cell(2, 0, true);
        occ_map.update_cell(2, 2, true);
        assert!(plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.0).is_some());
        assert!(plan_path_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.05).is_none());
    }
}
