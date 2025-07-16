function export_images_to_csv(folder_path, output_file, label)
  X = build_X(folder_path);
  % Build the matrix X from the images in folder_path
  X = X';
  % Transpose X to have images as rows
  m = size(X, 1);
  % m = number of images
  y = ones(m, 1) * label;
  % Create a label vector y with the specified label for each image
  % label = 1 for the folder with cats, label = 0 for the folder without cats
  data = [X y];
  % Combine the image data and labels into a single matrix

  if exist(output_file, 'file')
	% Read the existing data from the CSV file
    existing = csvread(output_file);
	% Append the new data to the existing data
    data = [existing; data];
  end


  csvwrite(output_file, data);
  % Write the combined data to the CSV file
end
