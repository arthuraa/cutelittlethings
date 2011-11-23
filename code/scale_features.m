function scale = scale_features(X, weights)
% SCALE_FEATURES -
%

row_weights = sum(X, 2);
for i = find(row_weights == 0)
    row_weights(i) = 1;
end
X = bsxfun(@rdivide, X', row_weights')';
X = bsxfun(@times, X', weights)';

scale = X;
