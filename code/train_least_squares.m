function model = train_least_squares(X, Y, quiet)
% TRAIN_LEAST_SQUARES -
%

if ~exist('quiet', 'var')
    quiet = false;
end

Xtrans = zeros(size(X));
coeffs = zeros(2, size(X, 2));

if ~quiet
    fprintf('Transforming feature matrix...\n');
    t = CTimeleft(size(X, 2));
end
for f = 1:size(X, 2)
    if ~quiet
        t.timeleft();
    end
    coeffs(1, f) = mean(Y(X(:, f) == 0));
    Xtrans(X(:, f) == 0, f) = coeffs(1, f);
    coeffs(2, f) = mean(Y(X(:, f) == 0));
    Xtrans(X(:, f) > 0, f) = coeffs(2, f);
end

Xtrans = X;

n_feat = size(X, 2);
program.C = [Xtrans ones(size(Xtrans, 1), 1)];
program.d = Y;
program.lb = -Inf;
program.up = Inf;
program.solver = 'lsqlin';
program.options = optimset('Display', 'off');

[alpha, resnorm] = lsqlin(program);
rmse = sqrt(resnorm / numel(Y));

model.alpha = alpha;
model.coeffs = coeffs;
model.rmse = rmse;


