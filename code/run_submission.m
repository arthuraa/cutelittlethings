%% Example submission: Naive Bayes

%% Load the data
load ../data/data_no_bigrams.mat;

%% Make the training data
% X = make_sparse(train);
% Y = double([train.rating]');

load XY_sparse.mat ; %% load X Y;
load Xtest_sparse ;
%% Make the testing data 
% save Xtest_sparse.mat Xtest
% Xtest = make_sparse(test); 


%% Run training
% Yk = bsxfun(@eq, Y, [1 2 4 5]);
% nb = nb_train_pk([X]'>0, [Yk]);

% %% Make the testing data and run testing
% Xtest = make_sparse(test, size(X, 2));
% Yhat = nb_test_pk(nb, Xtest'>0);

% %% Make predictions on test set

% % Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
% [tmp, Yhat] = max(Yhat, [], 2);
% ratings = [1 2 4 5];
% Yhat = ratings(Yhat)';
% save('-ascii', 'submit.txt', 'Yhat');


% print -djpeg -r72 naive_bayes.jpg;


%% liblinear
% which -all train
addpath ./liblinear-1.8/;
addpath ./liblinear-1.8/matlab ; 

%% scale 
format long 
XScale = bsxfun(@rdivide, X, max(X,[], 2));
Xtest_scale = bsxfun(@rdivide, Xtest, max(Xtest, [], 2)); 

% pretty slow 2 minutes, here bsxfun is not interesting 
% a lot of memeory consumed

%% train 
[test_err info ] = kernel_libsvm(Y,XScale);
% test_err =   0.108744483917733


%% predict 

%% get result 
[yhat, ~, vals] = predict(zeros(size(Xtest_scale,1),1), ... 
                          Xtest_scale,info.model); % the
                                                   % accuracy is not
                                                   % important
                                                   % for this model 
                                                   % yhat will pick
                                                   % the biggest
                                                   % value out of
                                                   % vals ;
save('-ascii', 'submit.txt', 'yhat');





