function a = sigmoid(z)
  % Compute sigmoid function
  a = 1 ./ (1 + exp(-z));
  % This is a vectorized implementation
  % of the sigmoid function
  % ./ is a element-wise division
  % exp(-z) computes the exponential of -z element-wise
end
