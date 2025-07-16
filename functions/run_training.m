function run_training(cat_folder, non_cat_folder, output_csv, alpha, num_iters)

  % run_training: Trains a model to distinguish between images of cats and non-cats
  %   cat_folder: Path to the folder containing images of cats
  %   non_cat_folder: Path to the folder containing images of non-cats
  %   output_csv: Path to the output CSV file where image data and labels will be saved
  %   alpha: Learning rate for the training algorithm
  %   num_iters: Number of iterations for the training algorithm

  % Set the true labels for the images
  % 1 for cat images, 0 for non-cat images
  export_images_to_csv(cat_folder, output_csv, 1);
  export_images_to_csv(non_cat_folder, output_csv, 0);

  % get the characteristics of the images and labels
  % from the CSV file
  [X, y] = get_characteristics(output_csv);

  % Compute w and b using the training algorithm
  [w, b, J] = train_model(X, y, alpha, num_iters);

  % y_pred = the result of prediction on the training set
  y_pred = predict(X, w, b);
  % Calculate the accuracy of the model
  accuracy = mean(double(y_pred == y)) * 100;

  fprintf('Final cost: %.4f\n', J);
  fprintf('Training accuracy: %.2f%%\n', accuracy);
end
