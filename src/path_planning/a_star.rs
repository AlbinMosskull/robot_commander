use crate::occupancy_map::occupancy_map::OccupancyMap;

fn hello_world() {
    let occ_map = OccupancyMap::new(3, 3);
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