function predictions = predict_gaussian(model, X, quiet)
% PREDICT_GAUSSIAN -
%
%   MODEL - model trained with train_gaussian
%   X - matrix of test points
%   QUIET - suppress output
%

densities = zeros(size(X, 1), 4);

for c = 1:4
    if ~(exist('quiet', 'var') && quiet == 'quiet');
        fprintf('Probabilities for c = %d...\n', c);
    end
    r = mvnpdf(X, model(c).mean, model(c).cov);
    for i = 1:numel(r)
        densities(i, c) = r(i);
    end
end

punctual = bsxfun(@times, [model.p], densities);
punctual = bsxfun(@times, 1./sum(punctual, 2), punctual);

predictions = punctual * [1;2;4;5];

% replace NaN entries with average
predictions(isnan(predictions)) = [model.p] * [1;2;4;5];
