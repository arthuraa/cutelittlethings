function model = train_gaussian(X, Y, C, quiet)
% TRAIN_GAUSSIAN - Mixture of gaussians model
%
%   X - N x K matrix of N training examples
%   Y - N x 1 vector of training labels
%   C - regularization constant
%   QUIET - suppress output
%
% OUTPUT:
%   MODEL.MEAN = gaussian means
%   MODEL.COV = covariance matrices
%   MODEL.P = class probability

a = [1;2;4;5];
for c = 1:4
    if ~(exist('quiet', 'var') && quiet == 'quiet')
        fprintf('Rating %d...\n', a(c));
    end
    class_idx = Y == a(c);
    Xclass = X(class_idx, :);

    % Regularize the data, force PD
    Xclass = vertcat(Xclass, eye(size(Xclass, 2)));

    model(c).mean = full(mean(Xclass));
    model(c).cov = full(cov(Xclass));
    model(c).p = sum(class_idx) / size(X, 1);

    % Check if covariance matrix is PD
    if ~test_pd(model(c).cov)
        fprintf('Warning: not PD\n');
    end
end
