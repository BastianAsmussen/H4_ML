use super::errors::NumberError as Error;

pub type Matrix<T> = Vec<Vec<T>>;

trait MatrixExt<T> {
    fn transpose(&self) -> Result<Matrix<T>, Error>;
}

impl<T: Copy + Default> MatrixExt<T> for Matrix<T> {
    fn transpose(&self) -> Result<Self, Error> {
        let v = self
            .first()
            .ok_or_else(|| Error::MatrixError("matrix is empty".to_string()))?;

        let mut transpose = vec![vec![T::default(); self.len()]; v.len()];
        for (i, row) in self.iter().enumerate() {
            for (j, col) in row.iter().enumerate() {
                transpose[j][i] = *col;
            }
        }

        Ok(transpose)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_matrix() {
        let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];

        let expected = vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]];
        let actual = matrix.transpose().expect("a valid matrix");

        assert_eq!(expected, actual);
    }
}
