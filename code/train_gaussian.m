function model = train_gaussian(X, Y, C, quiet)
% TRAIN_GAUSSIAN -
%

a = [1;2;4;5];
for c = 1:4
    if ~(exist('quiet', 'var') && quiet == 'quiet')
        fprintf('Rating %d...\n', a(c));
    end
    class_idx = Y == a(c);
    Xclass = X(class_idx, :);

    Xclass = vertcat(Xclass, eye(size(Xclass, 2)));

    model(c).mean = full(mean(Xclass));
    model(c).cov = full(cov(Xclass));
    model(c).p = sum(class_idx) / size(X, 1);

    % Check
    if ~test_pd(model(c).cov)
        fprintf('Warning: not PD\n');
    end
end
