function predictions = predict_least_squares(model, X)
% PREDICT_LEAST_SQUARES -
%

Xtrans = zeros(size(X));

for f = 1:size(X, 2)
    Xtrans(X(:, f) == 0) = model.coeffs(1, f);
    Xtrans(X(:, f) > 0) = model.coeffs(2, f);
end

Xtrans = X;

predictions = min(max([Xtrans ones(size(Xtrans, 1), 1)] * model.alpha, 1), 5);

