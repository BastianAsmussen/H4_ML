#[derive(Debug)]
pub struct Matrix(pub Vec<Vec<f64>>);

#[derive(Debug)]
pub struct LinearRegression {
    x: Matrix,
    y: Vec<f64>,
    alpha: f64,
}

impl LinearRegression {
    #[must_use]
    pub const fn new(x: Matrix, y: Vec<f64>, alpha: f64) -> Self {
        Self { x, y, alpha }
    }

    /*
    Algorithm: Linear Regression with Gradient Descent

    Input:
        X       // matrix of input features (size: m x n, where m = number of samples, n = number of features)
        y       // vector of target values (size: m)
        alpha   // learning rate
        N       // number of iterations

    Output:
        w       // weight vector (size: n)
        b       // bias term

    Initialize:
        w ← vector of zeros (size: n)
        b ← 0

    For i from 1 to N do:
        // Compute predictions for all samples
        y_pred ← X · w + b

        // Calculate the error between predictions and actual target values
        error ← y_pred - y

        // Compute gradients for weights and bias
        grad_w ← (1/m) * (transpose(X) · error)
        grad_b ← (1/m) * (sum of all elements in error)

        // Update the parameters
        w ← w - alpha * grad_w
        b ← b - alpha * grad_b

    Return w, b
    */
    fn gradient_descent(&self, n: usize) -> (Vec<f64>, f64) {
        let mut weights = Vec::with_capacity(n);
        let mut bias = 0.0;

        for i in 0..n {
            let y_pred = &self.x;
        }

        (weights, bias)
    }
}
