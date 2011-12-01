%% Load the data
load ../data/data_with_bigrams.mat;

%% make feature matrices

Y = double([train.rating])';

Xtrain_body = make_sparse(train, numel(vocab));
Xtrain_title = make_sparse_title(train, numel(vocab));
Xtrain_bigram = make_sparse_bigram(train, numel(bigram_vocab));
Xtrain_helpful = extract_helpful(train);
Xtrain_helpful_ratio = Xtrain_helpful(:,1) ./ (Xtrain_helpful(:,2) ...
                                               + 0.01);
Xtrain_length = sum(Xtrain_body, 2);

Xtest_body = make_sparse(test, numel(vocab));
Xtest_title = make_sparse_title(test, numel(vocab));
Xtest_bigram = make_sparse_bigram(test, numel(bigram_vocab));
Xtest_helpful = extract_helpful(test);
Xtest_helpful_ratio = Xtest_helpful(:,1) ./ (Xtest_helpful(:,2) ...
                                             + 0.01);
Xtest_length = sum(Xtest_body, 2);

%% compute histogram

% We remove features that occur more often in few categories than
% on the entire data set.

histogram.body = sum(Xtrain_body > 0);
histogram.title = sum(Xtrain_title > 0);
histogram.bigram = sum(Xtrain_bigram > 0);
categories = [train.category];
n_cats = max(categories);
select_body = ones(1, numel(vocab));
select_title = ones(1, numel(vocab));
select_bigram = ones(1, numel(bigram_vocab));

for c = 1:n_cats
    cat_hist_body = sum(Xtrain_body(categories == c, :) > 0);
    select_body((cat_hist_body / sum(categories == c)) ./ ...
                (histogram.body / numel(Y)) > 1.9) = 0;
    cat_hist_title = sum(Xtrain_title(categories == c, :) > 0);
    select_title((cat_hist_title / sum(categories == c)) ./ ...
                (histogram.title / numel(Y)) > 1.9) = 0;
    cat_hist_bigram = sum(Xtrain_bigram(categories == c, :) > 0);
    select_bigram((cat_hist_bigram / sum(categories == c)) ./ ...
                (histogram.bigram / numel(Y)) > 1.9) = 0;
end

%% select features

frequent_body_features = histogram.body > 100;
frequent_title_features = histogram.title > 100;
frequent_bigram_features = histogram.bigram > 100;

body_features = frequent_body_features & select_body;
title_features = frequent_title_features & select_title;
bigram_features = frequent_bigram_features & select_bigram;

train_features = [Xtrain_body(:, body_features) ...
                  Xtrain_title(:, title_features) ...
                  Xtrain_bigram(:, bigram_features) ...
                  Xtrain_helpful Xtrain_helpful_ratio ...
                  Xtrain_length];

test_features = [Xtest_body(:, body_features) ...
                 Xtest_title(:, title_features) ...
                 Xtest_bigram(:, bigram_features) ...
                 Xtest_helpful Xtest_helpful_ratio ...
                 Xtest_length];

fprintf('Using %d features\n', size(train_features, 2));

%% category cross validation

rmses = zeros(n_cats, 1);
for i = 1:n_cats
    fprintf('Category %d\n', i);
    fprintf('    Training model...\n');
    train_set = train_features(categories ~= i, :);
    train_labels = Y(categories ~= i, :);
    model = train_least_squares(train_set, train_labels, 1);
    pred = predict_least_squares(model, train_set);
    train_rmse = norm(pred - train_labels) / sqrt(size(train_labels, ...
                                                      1));
    fprintf('    Training RMSE = %f\n', train_rmse);
    fprintf('    Predicting on test set...\n');
    test_set = train_features(categories == i, :);
    test_labels = Y(categories == i, :);
    pred = predict_least_squares(model, test_set);
    rmses(i) = norm(pred - test_labels) / sqrt(size(test_labels, 1));
    fprintf('    Test RMSE = %f\n', rmses(i));
end

% FIXME: maybe this shouldn't be an even mean.
rmse = mean(rmses);
fprintf('Mean RMSE = %f\n', rmse);

%% train final model

model = train_least_squares(train_features, Y);

%% predict on test set

predictions = predict_least_squares(model, test_features);

save('-ascii', 'submit.linear-regression.txt', 'predictions');
