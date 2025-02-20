use std::fmt::Display;

pub mod linear_regression;

#[derive(Debug)]
pub enum Error {
    BadData(String),
}

impl Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadData(e) => write!(f, "Bad data error: {e}"),
        }
    }
}
