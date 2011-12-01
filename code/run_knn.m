%% load path

addpath ./kdtree;

%% load the data

load ../data/data_no_bigrams.mat;

%% build feature matrices

X.body = make_sparse(train, size(vocab, 2));
select_body = sum(X.body > 0) > 4000;
X.title = make_sparse_title(train, size(vocab, 2));
select_title = sum(X.title > 0) > 2000;
X.helpful = extract_helpful(train);
X.features = full([X.body(:, select_body) ...
                   X.title(:, select_title) ...
                   X.helpful]); % using a sparse matrix was causing
                                % the mex file to crash

fprintf('Selected %d features\n', size(X.features, 2));
Y = double([train.rating]');

categories = [train.category];

Xtest.body = make_sparse(test, size(vocab, 2));
Xtest.title = make_sparse_title(test, size(vocab, 2));
Xtest.helpful = extract_helpful(test);
Xtest.features = full([Xtest.body(:, select_body) ...
                    Xtest.title(:, select_title) ...
                    Xtest.helpful]);

%% build tree

model = train_knn(X.features, Y, 100, 10);

%% train error

p = predict_knn(model, X.features, 2);

%% free tree

destroy_knn_model(model);
