use crate::numbers::matrix::Matrix;

#[derive(Debug)]
pub struct LinearRegression<T> {
    x: Matrix<T>,
    y: Vec<T>,
}

impl<T> LinearRegression<T> {
    #[must_use]
    pub const fn new(x: Matrix<T>, y: Vec<T>) -> Self {
        Self { x, y }
    }
}
