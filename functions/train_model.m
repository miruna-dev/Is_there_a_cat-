function [w, b, J] = train_model(X, y, alpha, num_iters)
  [m, n] = size(X);
  % Initialize weights and bias
  w = zeros(n, 1);
  b = 0;

  % Perform gradient descent to learn w and b
  [w, b] = gradient_descent(X, y, w, b, alpha, num_iters);

  % Compute the final cost
  z = X * w + b;
  a = sigmoid(z);
  % J shows how good is the prediction
  J = compute_cost(a, y);
end
