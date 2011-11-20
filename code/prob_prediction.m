function prediction = prob_prediction(probs, p)
% Make a prediction based on probabilities. PROBS is a N x 4 matrix
% of class probabilities on N test examples. P is a factor to
% "sharpen" the probabilities.

% Rescale probabilities
probs = probs.^p;
probs = bsxfun(@rdivide, probs, sum(probs, 2));

prediction = probs * [1; 2; 4; 5];
