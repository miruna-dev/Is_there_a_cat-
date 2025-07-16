% pkg load image
% Loading image package so we can use some matlab image functions

function [R, G, B] = image_to_rgb_matrix(image_path)
  % Read the image from the specified path
  img = imread(image_path);
  
  img = imresize(img, [64, 64]);
  % Resize the image to 64x64 pixels

  if size(img, 3) == 1
  % If the image is grayscale, convert it to RGB by duplicating the channel
  % This ensures that we have three channels for R, G, and B
    img = cat(3, img, img, img);
  end

  R = img(:,:,1);
  G = img(:,:,2);
  B = img(:,:,3);
  % Extract the red, green, and blue channels from the image

  % Each R, G and B will be a 64x64 matrix
  % representing the intensity of the respective color channel
  % in the image

end