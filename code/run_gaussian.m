%% loading

load ../data/data_with_bigrams.mat;

%% word frequency histogram

histogram_body = zeros(size(vocab, 2), 1);
histogram_title = zeros(size(vocab, 2), 1);
histogram_bigram = zeros(size(bigram_vocab, 2), 1);
for i = 1:size(train, 2)
    histogram_body(train(i).word_idx) = ...
        histogram_body(train(i).word_idx) + ...
        double(train(i).word_count);
    histogram_title(train(i).title_idx) = ...
        histogram_title(train(i).title_idx) + ...
        double(train(i).title_count);
    histogram_bigram(train(i).bigram_idx) = ...
        histogram_bigram(train(i).bigram_idx) + ...
        double(train(i).bigram_count);
end

include_for_body = histogram_body > 100;
include_for_title = histogram_title > 100;
include_for_bigram = histogram_bigram > 100;

%% build feature matrix with pre filtering

Xbody = make_sparse(train, size(vocab, 2));
Xtitle = make_sparse_title(train, size(vocab, 2));
Xbigram = make_sparse_bigram(train, size(bigram_vocab, 2));
Xhelpful = extract_helpful(train);
X = [Xbody(:,include_for_body) Xtitle(:,include_for_title) ...
     Xbigram(:, include_for_bigram) Xhelpful];
Y = double([train.rating]');

Xtest_body = make_sparse(test, size(vocab, 2));
Xtest_title = make_sparse_title(test, size(vocab, 2));
Xtest_bigram = make_sparse_bigram(test, size(bigram_vocab, 2));
Xtest_helpful = extract_helpful(test);
Xtest = [Xtest_body(:,include_for_body) ...
         Xtest_title(:,include_for_title) ...
         Xtest_bigram(:,include_for_bigram) ...
         Xtest_helpful];

%% select good features

features = select_features_gaussian(X, Y, 400);
Xtrain = X(:, features);
Xtest = Xtest(:, features);

%% train model

model = train_gaussian(Xtrain, Y, 0.0025);

% training RMSE

training_predictions = predict_gaussian(model, Xtrain);
training_rmse = norm(Y - training_predictions) / sqrt(size(Y, 1))

% 1000, 500 = 1.5660
% 1000, 100 = 1.2687
% 1000, 100 + helpful = 1.2647

%% predictions

predictions = predict_gaussian(model, Xtest);

save('-ascii', 'submit.gaussian.txt', 'predictions');
