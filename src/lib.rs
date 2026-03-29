mod core;
mod occupancy_map;
mod path_planning;

use pyo3::prelude::*;                                                      
use occupancy_map::occupancy_map::OccupancyMap;

#[pymodule]                                                                
fn robot_commander(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OccupancyMap>()?;                                        
    Ok(())      
}