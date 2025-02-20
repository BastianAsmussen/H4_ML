use std::fmt::Display;

#[derive(Debug)]
pub enum NumberError {
    MatrixError(String),
}

impl Display for NumberError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MatrixError(e) => write!(f, "matrix error: {e}")
        }
    }
}
