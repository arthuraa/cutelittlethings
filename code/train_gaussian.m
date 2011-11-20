function model = train_gaussian(X, Y, C, quiet)
% TRAIN_GAUSSIAN -
%

a = [1;2;4;5];
for c = 1:4
    if ~(exist('quiet', 'var') && quiet ~= 'quiet')
        fprintf('Rating %d...\n', c);
    end
    class_idx = Y == a(c);
    Xclass = X(class_idx, :);

    % force positive definite
    ex = (sum(Xclass) == 0) * C;
    Xclass = vertcat(Xclass, ex);

    model(c).mean = mean(Xclass);
    model(c).cov = cov(Xclass);
    model(c).p = sum(class_idx) / size(X, 1);
end
