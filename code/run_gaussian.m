%% loading

load ../data/data_no_bigrams.mat;

%% word frequency histogram

histogram_body = zeros(size(vocab, 2), 1);
histogram_title = zeros(size(vocab, 2), 1);
for i = 1:size(train, 2)
    histogram_body(train(i).word_idx) = ...
        histogram_body(train(i).word_idx) + ...
        double(train(i).word_count);
    histogram_title(train(i).title_idx) = ...
        histogram_title(train(i).title_idx) + ...
        double(train(i).title_count);
end

include_for_body = histogram_body > 1000;
include_for_title = histogram_title > 100;

%% build feature matrix, selecting important features

Xbody = make_sparse(train, size(vocab, 2));
Xtitle = make_sparse_title(train, size(vocab, 2));
X = [Xbody Xtitle];
X = X(:, vertcat(include_for_body, include_for_title));

Xtest_body = make_sparse(test, size(vocab, 2));
Xtest_title = make_sparse_title(test, size(vocab, 2));
Xtest = [Xtest_body Xtest_title];
Xtest = Xtest(:, vertcat(include_for_body, include_for_title));

%% train model

model = train_gaussian(X, [train.rating], 0.0025);

% training RMSE

training_predictions = predict_gaussian(model, X);
training_rmse = norm(double([train.rating]') - training_predictions) / ...
    sqrt(size(training_predictions, 1))

% 1000, 500 = 1.5660
% 1000, 100 = 1.2687




%% predictions

predictions = predict_gaussian(model, Xtest);
