function corrs = rating_corr(X, Y)
% RATING_CORR - calculate the correlation between input features and
% review ratings.
%

corrs = zeros(1, size(X, 2));

t = CTimeLeft(size(X, 2));
for i = 1:size(X, 2)
    t.timeleft();
    m = corrcoef(X(:,i), Y);
    corrs(i) = m(1,2);
end
