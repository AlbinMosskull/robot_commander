pub fn multiply(a: i32, b: i32) -> i32 {
    let mut sum = 0;
    for _ in 0..b {
        sum += a;
    }
    sum
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiply_basic_test() {
        let result = multiply(2, 3);
        assert_eq!(result, 6);
    }

    #[test]
    fn multiply_negative() {
        let result = multiply(-2, 1);
        assert_eq!(result, -2);
    }

    #[test]
    fn multiply_negative_opposite_order() {
        let result = multiply(1, -2);
        assert_eq!(result, -2);
    }
}
