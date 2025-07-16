function y_pred = predict(X, w, b)
  z = X * w + b;
  a = sigmoid(z);
  y_pred = a >= 0.5;
end
