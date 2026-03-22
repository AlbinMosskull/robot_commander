
pub struct MinHeap {
    heap: Vec<i32>,
}

impl MinHeap {
    pub fn new() -> Self {
        MinHeap {
            heap: Vec::new()
        }
    }

    pub fn extract_min(&mut self) -> Option<i32> {
        if self.len() == 0 {
            return None;
        }

        let min_element = self.heap[0];
        self.sink_down();
        Some(min_element)
    }

    pub fn insert(&mut self, element: i32) {
        self.heap.push(element);
        self.bubble_up();
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn simple_print(&self) {
        for el in &self.heap {
            print!("{} ", el);
        }
        println!("");
    }

    fn bubble_up(&mut self) {
        let mut current_index = self.len() - 1;

        while current_index != 0 {
            let parent: usize = (current_index - 1) / 2;
            let current_element = self.heap[current_index];
            let parent_element = self.heap[parent];

            if parent_element > current_element {
                self.heap.swap(current_index, parent);
                current_index = parent;
                continue;
            } else {
                return;
            }

        }
    }

    fn sink_down(&mut self) {
        if self.len() == 1 {
            self.heap.pop();
            return;
        }

        let last_element = self.heap.pop();
        self.heap[0] = last_element.expect("Given the prior checks, should always have a value here");
        let full_length = self.len();

        let mut current_index = 0;
        loop {
            let current_element = self.heap[current_index];

            let left_child = current_index * 2 + 1;
            let right_child = current_index * 2 + 2;

            // Find smallest child

            if left_child >= full_length {
                return;
            } 
            let left_child_element = self.heap[left_child];
            if left_child_element < current_element {
                if right_child >= full_length {
                    self.heap.swap(current_index, left_child);
                    return;
                } else {
                    let right_child_element = self.heap[right_child];

                    if right_child_element < left_child_element {
                        self.heap.swap(current_index, right_child);
                        current_index = right_child;
                    } else {
                        self.heap.swap(current_index, left_child);
                        current_index = left_child;
                    }
                    continue;
                }

            }

            if right_child >= full_length {
                return;
            }
            let right_child_element = self.heap[right_child];
            if right_child_element < current_element {
                self.heap.swap(current_index, right_child);
                current_index = right_child;
                continue;
            }
            
            // If left child and right child are bigger or equal to current_element (the parent)
            // we are done.
            break;
        }

    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_test() {
        let mut min_heap = MinHeap{heap: vec![1; 3]};
        min_heap.insert(2);
        min_heap.insert(3);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 1);
        assert_eq!(min_heap.len(), 4);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 1);
        assert_eq!(min_heap.len(), 3);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 1);
        assert_eq!(min_heap.len(), 2);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 2);
        assert_eq!(min_heap.len(), 1);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 3);
    }

    #[test]
    fn basic_test_2() {
        let mut min_heap = MinHeap::new();
        min_heap.insert(4);
        min_heap.insert(5);
        min_heap.insert(1);
        min_heap.insert(3);
        min_heap.insert(2);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 1);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 2);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 3);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 4);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 5);
    }

    #[test]
    fn extract_min_empty_heap() {
        let mut min_heap = MinHeap::new();
        assert_eq!(min_heap.extract_min(), None);
    }

    #[test]
    fn sink_down_smallest_child_swapped() {
        let mut min_heap = MinHeap::new();
        min_heap.insert(1);
        min_heap.insert(2);
        min_heap.insert(5);
        assert_eq!(min_heap.heap[0], 1);
    }

    #[test]
    fn sink_down_equal_elements() {
        let mut min_heap = MinHeap::new();
        min_heap.insert(1);
        min_heap.insert(1);
        min_heap.insert(1);
        min_heap.insert(1);
        min_heap.insert(1);
        min_heap.insert(1);
        min_heap.insert(1);
        assert_eq!(min_heap.extract_min().expect("Should have a value"), 1);

    }
    
}