use pyo3::prelude::*;

#[pyclass]
#[derive(Copy, Clone)]
pub struct WorldPosition2d {
    #[pyo3(get)]
    pub x: f32,
    #[pyo3(get)]
    pub y: f32,
}

#[pymethods]
impl WorldPosition2d {
    #[new]
    pub fn new(x: f32, y: f32) -> Self {
        WorldPosition2d { x, y }
    }
}
