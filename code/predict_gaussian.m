function predictions = predict_gaussian(model, X, quiet)
% PREDICT_GAUSSIAN -
%

% E[Y|X = x] = \sum

densities = zeros(size(X, 1), 4);

for c = 1:4
    if ~(exist('quiet', 'var') && quiet == 'quiet');
        fprintf('Probabilities for c = %d...\n', c);
    end
    densities(:, c) = mvnpdf(X, model(c).mean, model(c).cov);
end

punctual = bsxfun(@times, [model.p], densities);
punctual = bsxfun(@times, 1./sum(punctual, 2), punctual);

predictions = punctual * [1;2;4;5];

% replace NaN entries with average
predictions(isnan(predictions)) = [model.p] * [1;2;4;5];