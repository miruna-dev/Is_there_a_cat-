function [X, y] = get_characteristics(csv_file)

  % This function reads a CSV file and returns the features and labels.
  data = dlmread(csv_file, ',');
  % The last column is the label
  X = data(:, 1:end-1);
  y = data(:, end);
  % Normalize the features
  X = (X - min(X)) ./ (max(X) - min(X));
end
