mod core;
mod occupancy_map;
mod path_planning;

use pyo3::prelude::*;                                                      
use core::geometry_types::WorldPosition2d;
use occupancy_map::occupancy_map::OccupancyMap;
use path_planning::a_star::plan_path;
use path_planning::a_star::plan_path_towards_goal;

#[pymodule]                                                                
fn robot_commander(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OccupancyMap>()?;                                        
    m.add_class::<WorldPosition2d>()?;                                        
    m.add_function(wrap_pyfunction!(plan_path, m)?)?;
    m.add_function(wrap_pyfunction!(plan_path_towards_goal, m)?)?;
    Ok(())      
}