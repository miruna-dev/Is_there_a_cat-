function [w, b] = gradient_descent(X, y, w, b, alpha, num_iters)

  % Perform gradient descent to learn w and b
  % X = input features
  % y = true labels (0 or 1)
  % w = weights (initial guess)
  % b = bias (initial guess)
  % alpha = learning rate
  % num_iters = number of iterations for gradient descent
  % m = number of training examples
  m = length(y);

  for i = 1:num_iters
  % Each step we update the weights and bias in the direction where the cost decreases
    z = X * w + b;
	% a = the probability to be a cat in the given image
    a = sigmoid(z);
	% Compute the error between predicted and true labels
    dz = a - y;
	% Compute gradients for weights and bias
    dw = (1/m) * (X' * dz);
    db = (1/m) * sum(dz);
	% Update weights and bias to minimize the cost
    w = w - alpha * dw;
    b = b - alpha * db;
  end
end
