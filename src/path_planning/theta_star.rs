use std::collections::HashMap;
use std::cmp::Ordering;

use line_drawing::bresenham;
use pyo3::prelude::*;

use crate::occupancy_map::occupancy_map::OccupancyMap;
use crate::occupancy_map::occupancy_map::Position2d;
use crate::core::min_heap::MinHeap;
use crate::core::geometry_types::WorldPosition2d;


#[pyfunction]
pub fn plan_path_theta_star(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Option<Vec<WorldPosition2d>> {
    let start = occ_map.convert_coordinate_to_index(world_start.x, world_start.y)?;
    let goal = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y)?;

    let positions = plan_theta_star_indices(occ_map, start, goal, collision_margin)?;
    Some(positions.into_iter().map(|p| occ_map.convert_index_to_coordinate(p)).collect())
}

#[pyfunction]
pub fn plan_path_towards_goal_theta_star(occ_map: &OccupancyMap, world_start: WorldPosition2d, world_goal: WorldPosition2d, collision_margin: f32) -> Vec<WorldPosition2d> {
    let start = occ_map.convert_coordinate_to_index(world_start.x, world_start.y).expect("Must give a position within world bounds");
    let goal = occ_map.convert_coordinate_to_index(world_goal.x, world_goal.y).expect("Must give a position within world bounds");

    let positions = plan_towards_goal_theta_star_indices(occ_map, start, goal, collision_margin);
    positions.into_iter().map(|p| occ_map.convert_index_to_coordinate(p)).collect()
}

fn plan_theta_star_indices(occ_map: &OccupancyMap, start: Position2d, goal: Position2d, collision_margin: f32) -> Option<Vec<Position2d>> {
    let result = plan_towards_goal_theta_star_indices(occ_map, start, goal, collision_margin);
    (result.last() == Some(&goal)).then_some(result)
}

fn plan_towards_goal_theta_star_indices(occ_map: &OccupancyMap, start: Position2d, goal: Position2d, collision_margin: f32) -> Vec<Position2d> {
    let mut came_from: HashMap<Position2d, Position2d> = HashMap::new();
    let mut best_g_scores: HashMap<Position2d, f32> = HashMap::new();
    let mut open_set: MinHeap<Position2dWithCost> = MinHeap::new();

    open_set.insert(Position2dWithCost {
        position: start,
        g_cost: 0.0,
        h_cost: euclidean_distance(start, goal),
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

        let current_g = best_g_scores.get(&current).copied().expect("Should always have a value");
        let grandparent = came_from.get(&current).copied();

        for neighbor in get_neighbors(occ_map, current, collision_margin) {
            let neighbor_best_g = best_g_scores.get(&neighbor).copied().unwrap_or(f32::INFINITY);

            let (tentative_parent, tentative_g) = match grandparent {
                Some(grandparent) if has_line_of_sight(occ_map, grandparent, neighbor, collision_margin) => {
                    let grandparent_g = best_g_scores.get(&grandparent).copied().expect("Should always have a value");
                    (grandparent, grandparent_g + euclidean_distance(grandparent, neighbor))
                }
                _ => (current, current_g + euclidean_distance(current, neighbor)),
            };

            if tentative_g < neighbor_best_g {
                came_from.insert(neighbor, tentative_parent);
                best_g_scores.insert(neighbor, tentative_g);
                open_set.insert(Position2dWithCost {
                    position: neighbor,
                    g_cost: tentative_g,
                    h_cost: euclidean_distance(neighbor, goal),
                });
            }
        }
    }

    let closest_to_goal = *best_g_scores.keys()
        .min_by(|a, b| euclidean_distance(**a, goal).partial_cmp(&euclidean_distance(**b, goal)).unwrap())
        .expect("best_g_scores should never be empty");
    reconstruct_path(&came_from, closest_to_goal)
}


#[derive(Copy, Clone)]
struct Position2dWithCost {
    position: Position2d,
    g_cost: f32,
    h_cost: f32,
}

impl Position2dWithCost {
    fn total_cost(&self) -> f32 {
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

fn reconstruct_path(came_from: &HashMap<Position2d, Position2d>, final_position: Position2d) -> Vec<Position2d> {
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

fn get_neighbors(occ_map: &OccupancyMap, position: Position2d, collision_margin: f32) -> Vec<Position2d> {
    const DIRECTIONS: [(isize, isize); 8] = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ];
    DIRECTIONS.iter()
        .map(|(dx, dy)| Position2d { x: position.x + dx, y: position.y + dy })
        .filter(|neighbor| occ_map.is_valid_index(*neighbor, collision_margin))
        .collect()
}

fn euclidean_distance(a: Position2d, b: Position2d) -> f32 {
    let dx = (b.x - a.x) as f32;
    let dy = (b.y - a.y) as f32;
    (dx * dx + dy * dy).sqrt()
}

fn has_line_of_sight(occ_map: &OccupancyMap, from: Position2d, to: Position2d, collision_margin: f32) -> bool {
    bresenham((from.x, from.y), (to.x, to.y))
        .into_iter()
        .all(|(x, y)| occ_map.is_valid_index(Position2d { x, y }, collision_margin))
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
        let path = plan_theta_star_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0).expect("Should find a path");
        assert!(!path.is_empty());
        assert_eq!(*path.first().unwrap(), Position2d { x: 0, y: 0 });
        assert_eq!(*path.last().unwrap(), Position2d { x: 2, y: 2 });
    }

    #[test]
    fn produces_shorter_path_than_a_star_in_open_space() {
        // In open space, Theta* connects start directly to goal via line-of-sight, giving a 2-node path.
        // A* with 4-connectivity produces 5 nodes for the same (0,0) -> (2,2) query.
        let occ_map = make_map(5, 5);
        let path = plan_theta_star_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 4, y: 4 }, 0.0).expect("Should find a path");
        assert_eq!(path.len(), 2);
    }

    #[test]
    fn start_equals_goal_returns_single_node_path() {
        let occ_map = make_map(5, 5);
        let path = plan_theta_star_indices(&occ_map, Position2d { x: 2, y: 2 }, Position2d { x: 2, y: 2 }, 0.0).expect("Should find a path");
        assert_eq!(path.len(), 1);
    }

    #[test]
    fn path_routes_around_obstacle() {
        let mut occ_map = make_map(5, 3);
        occ_map.update_cell(2, 0, true);
        occ_map.update_cell(2, 1, true);
        let path = plan_theta_star_indices(&occ_map, Position2d { x: 0, y: 1 }, Position2d { x: 4, y: 1 }, 0.0).expect("Should route around obstacle");
        assert_eq!(*path.first().unwrap(), Position2d { x: 0, y: 1 });
        assert_eq!(*path.last().unwrap(), Position2d { x: 4, y: 1 });
    }

    #[test]
    fn returns_none_when_goal_is_unreachable() {
        let mut occ_map = make_map(3, 3);
        occ_map.update_cell(1, 2, true);
        occ_map.update_cell(2, 1, true);
        occ_map.update_cell(1, 1, true);
        occ_map.update_cell(2, 2, true);
        let path = plan_theta_star_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0);
        assert!(path.is_none());
    }

    #[test]
    fn towards_goal_reaches_goal_when_reachable() {
        let occ_map = make_map(5, 5);
        let path = plan_towards_goal_theta_star_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 4, y: 4 }, 0.0);
        assert_eq!(*path.last().unwrap(), Position2d { x: 4, y: 4 });
    }

    #[test]
    fn towards_goal_ends_at_closest_reachable_node_when_goal_blocked() {
        let mut occ_map = make_map(3, 3);
        // Block the goal and all its 8-connected neighbors so it is truly unreachable
        occ_map.update_cell(1, 1, true);
        occ_map.update_cell(1, 2, true);
        occ_map.update_cell(2, 1, true);
        occ_map.update_cell(2, 2, true);
        let path = plan_towards_goal_theta_star_indices(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 2, y: 2 }, 0.0);
        assert!(!path.is_empty());
        assert_ne!(*path.last().unwrap(), Position2d { x: 2, y: 2 });
    }

    #[test]
    fn line_of_sight_clear_in_open_map() {
        let occ_map = make_map(10, 10);
        assert!(has_line_of_sight(&occ_map, Position2d { x: 0, y: 0 }, Position2d { x: 9, y: 9 }, 0.0));
    }

    #[test]
    fn line_of_sight_blocked_by_wall() {
        let mut occ_map = make_map(10, 10);
        for y in 0..10 {
            occ_map.update_cell(5, y, true);
        }
        assert!(!has_line_of_sight(&occ_map, Position2d { x: 0, y: 5 }, Position2d { x: 9, y: 5 }, 0.0));
    }
}
