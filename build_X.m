function X = build_X(folder_path)
  % Function that builds the matrix X from the images in folder_path
  files = dir(fullfile(folder_path, '*.jpg')); 
  m = length(files);
  % m = number of images

  % Get the path of the first image to determine the size of the vectors
  first_image_path = fullfile(folder_path, files(1).name);
  [R, G, B] = image_to_rgb_matrix(first_image_path);
  % Read the first image and convert it to RGB matrices
  x1 = rgb_to_vector(R, G, B);
  % Convert RGB matrices to a single vector
  n = length(x1);
  % n = number of pixels in each image (64*64*3 = 12288)

  X = zeros(n, m);
  % Initialize X as a zero matrix of size n x m

  for i = 1:m
    % Loop through each image file
    % Read the image and convert it to RGB matrices
    img_path = fullfile(folder_path, files(i).name);
    [R, G, B] = image_to_rgb_matrix(img_path);
    % Convert RGB matrices to a single vector
    % and store it in the i-th column of X
    x = rgb_to_vector(R, G, B);
    X(:, i) = x;
  end
end
