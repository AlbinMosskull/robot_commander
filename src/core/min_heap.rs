
pub struct MinHeap {
    heap: Vec<i32>,
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_heap_dummy() {
        let mut min_heap = MinHeap{heap: vec![1; 1]};
        min_heap.heap.push(1);
        assert!(true);
    }

}