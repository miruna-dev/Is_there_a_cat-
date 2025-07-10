function x = rgb_to_vector(R, G, B)
  % Convert RGB matrices to a single vector
  x = [R(:); G(:); B(:)];
end
