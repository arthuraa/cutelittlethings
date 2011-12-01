function model = train_least_squares(X, Y, quiet)
% TRAIN_LEAST_SQUARES - Train a linear regression model
%
%   X - N x K feature matrix with N data points
%   Y - N x 1 vector of training labels
%   QUIET - Suppress output

if ~exist('quiet', 'var')
    quiet = false;
end

n_feat = size(X, 2);
program.C = [X ones(size(X, 1), 1)];
program.d = Y;
program.lb = -Inf;
program.up = Inf;
program.solver = 'lsqlin';
program.options = optimset('Display', 'off');

[alpha, resnorm] = lsqlin(program);
rmse = sqrt(resnorm / numel(Y));

model.alpha = alpha;
model.rmse = rmse;


