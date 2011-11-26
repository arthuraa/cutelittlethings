%% Load the data
% import test train vocab
format long ;
addpath ./liblinear-1.8/;
addpath ./liblinear-1.8/matlab ;

%% slow loading
load ../data/data_no_bigrams.mat;

%% make feature matrices

Y = double([train.rating])';

X = make_sparse(train, numel(vocab));
Xtrain_title = make_sparse_title(train, numel(vocab));
Xtrain_helpful = extract_helpful(train);
Xtrain_helpful_ratio = Xtrain_helpful(:,1) ./ (Xtrain_helpful(:,2) ...
                                               + 0.01);
Xtrain_length = sum(X, 2);

Xtest = make_sparse(test, numel(vocab));
Xtest_title = make_sparse_title(test, numel(vocab));
Xtest_helpful = extract_helpful(test);
Xtest_helpful_ratio = Xtest_helpful(:,1) ./ (Xtest_helpful(:,2) ...
                                             + 0.01);
Xtest_length = sum(Xtest, 2);


%Xtest = make_sparse(test, size(vocab, 2));
%Xtest_title = make_sparse_title(test, size(vocab, 2));

%% liblinear

%% scale
scale_words = max(max(X), 1); % weird out of memory bug
scale_title = max(max(Xtrain_title), 1);
scale_helpful = max(max(Xtrain_helpful), 1);
scale_length = max(Xtrain_length);

Xtrain_scale = bsxfun(@rdivide, X', scale_words')';
Xtrain_title_scale = bsxfun(@rdivide, Xtrain_title', ...
                            scale_title')';
Xtrain_helpful_scale = bsxfun(@rdivide, Xtrain_helpful', ...
                              scale_helpful')';
Xtrain_length_scale = Xtrain_length / scale_length;

Xtest_scale = bsxfun(@rdivide, Xtest', scale_words')';
Xtest_title_scale = bsxfun(@rdivide, Xtest_title', ...
                            scale_title')';
Xtest_helpful_scale = bsxfun(@rdivide, Xtest_helpful', ...
                              scale_helpful')';
Xtest_length_scale = Xtest_length / scale_length;

%% cross validation
categories_train = [train.category];
combine_features = [Xtrain_scale (1.149*Xtrain_title_scale) ...
                    (5.25*Xtrain_helpful_scale) ...
                    (5.25*Xtrain_helpful_ratio) ...
                    (8*Xtrain_length_scale)];
rmse = ...
    category_validation_all(Y, double(combine_features), ...
                            double(categories_train), ...
                            '-s 4 -c 0.068 -q', 9.09)

%% test using those parameters
[acc, info] = kernel_libsvm(Y, combine_features, '-s 4 -q', ...
                            [0.068]);
combine_test = [Xtest_scale (1.149*Xtest_title_scale) ...
                    (5.25*Xtest_helpful_scale) ...
                    (5.25*Xtest_helpful_ratio) ...
                    (8*Xtest_length_scale)];

Yhat = predict_aux(combine_test, info.model, 9.09);

save('-ascii', 'submit.txt', 'Yhat');

