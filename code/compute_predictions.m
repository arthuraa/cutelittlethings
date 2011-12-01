function predictions = compute_predictions(model, X, p)
% COMPUTE_PREDICTIONS -
%
[yhat, acc, probs] = predict(zeros(size(X, 1), 1), X, model, '-b 1');
a = probs.^p;
predictions = bsxfun(@rdivide,a,sum(a,2)) * [1;2;4;5];
