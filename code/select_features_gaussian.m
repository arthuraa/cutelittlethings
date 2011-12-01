function features = select_features_gaussian(X, Y, N)
% SELECT_FEATURES_GAUSSIAN -
%

corrs = rating_corr(X, Y);

sorted = sort(abs(corrs), 'descend');

cut = sorted(N);

features = abs(corrs) >= cut;
