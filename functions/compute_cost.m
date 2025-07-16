function J = compute_cost(a, y)
  % Compute the cost for logistic regression
  % m = number of training examples
  % a = predicted probabilities (output of the logistic function)
  % y = true labels (0 or 1)
  m = length(y);
  J = (-1/m) * sum(y .* log(a) + (1 - y) .* log(1 - a));
end
