function [h] = plot_dt_digit(t, x)
% PLOT_DT_DIGIT - Plots the pixels used by the decision tree.
%
% Usage:
%
%   H = PLOT_DT_DIGIT(T, X)
%
%  Returns the handle to an image plot. The image shows the handdrawn digit
%  in red, with the pixels chosen by the tree to evaluate the image
%  highlighted, colored by the order in which they were evaluated (bright
%  green = first, dark blue = last). Assumes T is the root node of a
%  decision tree, and X is a single 1 x D vector version of the imag.

% Evaluate the tree, but keep track of which features were used.
node = t;
fidx = [];
while ~node.terminal
    fidx = [fidx node.fidx];
    if x(node.fidx) <= node.fval
        node = node.left;
    else
        node = node.right;
    end
end
y = node.value;

% Convert the input feature vector into an RGB image, with the red channel
% greater than the others.
im = reshape(x, [28 28])';
im_rgb = repmat(im, [1 1 3]);
im_rgb(:,:,[2 3]) = im_rgb(:,:,[2 3]) .* 0.5;

% Generate an image with just the pixels that were chosen by the DT, so
% that they show up in the correct spot in the visualization.
x_f = zeros(size(x));
x_f(fidx) = 1:numel(fidx);
im_f = reshape(x_f, [28 28])';

% Find the row and col indices of the selected pixels.
[i j] = find(im_f);

% Prepare colors for the pixels (using scheme 'winter').
colors = winter(numel(fidx)).*255;
colors = colors(end:-1:1, :);

% Color the pixels in the letter image.
for k = 1:numel(i)
    im_rgb(i(k),j(k),:) = reshape(colors(im_f(i(k),j(k)),:), 1, 1, 3);
end

% Display image for the user with appropriate title.
h = image(uint8(im_rgb)); axis image; axis off;
title(['P(Y) = ' num2str(y, '%.2f,')]);