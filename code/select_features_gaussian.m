function features = select_features_gaussian(X, Y, N)
% SELECT_FEATURES_GAUSSIAN -
%

corrs = zeros(1, size(X, 2));

t = CTimeLeft(size(X, 2));
for i = 1:size(X, 2)
    % t.timeleft();
    m = corrcoef(X(:,i), Y);
    corrs(i) = m(1,2);
end

% sorted = sort(abs(corrs), 'descend');
% cut = sorted(N);

features = corrs;
