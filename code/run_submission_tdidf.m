%% Example submission: Naive Bayes

%% Load the data
% import test train vocab
format long ;
addpath ./liblinear-1.8/;
addpath ./liblinear-1.8/matlab ;

%% slow loading
load ../data/data_no_bigrams.mat;

%% quick loading
X = make_sparse(train, size(vocab, 2));
Y = double([train.rating])';
Xtest = make_sparse(test, size(vocab, 2));
load 'stop_index.txt' stop_index ; % load the data into stop_index;
load invalid_index.mat ;
idf_data = importdata('idf.txt', ' ');
word_weights = idf_data.data;
word_weights = word_weights / max(word_weights); % scale word weights


%% liblinear

%% scale
scale_word = max(max(X), 1);
Xtrain_scale = bsxfun(@rdivide, X',scale_word')';
Xtrain_scale = bsxfun(@times, Xtrain_scale', word_weights)';

% since we can only use the features we have
Xtest_scale = bsxfun(@rdivide, Xtest', scale_word')';
Xtest_scale = bsxfun(@times, Xtest_scale', word_weights)';

Xtrain_title= make_sparse_title(train, size(vocab, 2)) ;
scale_title = max(max(Xtrain_title),1);
Xtrain_title_scale = bsxfun(@rdivide, Xtrain_title', ...
                            scale_title')';
Xtrain_title_scale = bsxfun(@times, Xtrain_title_scale', word_weights)';
Xtest_title = make_sparse_title(test, size(vocab, 2));
Xtest_title_scale = bsxfun(@rdivide, Xtest_title', scale_title')';
Xtest_title_scale = bsxfun(@times, Xtest_title_scale', word_weights)';

%% cross validation
categories_train = [train.category];
combine_features = [Xtrain_scale (1.149*Xtrain_title_scale)];
rmse = ...
    category_validation_all(Y, double(combine_features), ...
                            double(categories_train), ...
                            '-s 4 -c 0.068 -q', 9.09)

% c = 0.06, p = 9, f = 1.1: 0.897435
% c = 0.08, p = 9, f = 1.1: 0.8996
% c = 0.06, p = 9, f = 1.1: 0.897412
% c = 0.07, p = 9, f = 1.1: 0.897392
% c = 0.069, p = 9, f = 1.1: 0.897375
% c = 0.068, p = 9, f = 1.1: 0.897356
% c = 0.068, p = 9, f = 1.15: 0.897335
% c = 0.068, p = 9.09, f = 1.149: 0.897321

%% test using those parameters
[acc, info] = kernel_libsvm(Y, combine_features, '-s 4 -q', ...
                            [0.068]);
combine_test = [Xtest_scale, (1.149*Xtest_title_scale)];

Yhat = predict_aux(combine_test, info.model, 9.09);

save('-ascii', 'submit.11-19.txt', 'Yhat');

%% title using bigram ?
Xtrain_bigram = make_sparse_bigram(train);
scale_bigram_word = max(max(Xtrain_bigram),1);
Xtrain_bigram_scale = bsxfun(@rdivide, ...
                             Xtrain_bigram,scale_bigram_word);
%% test bigram
Xtest_bigram = make_sparse_bigram(test);
Xtest_bigram = Xtest_bigram(:,1:size(Xtrain_bigram_scale,2));
Xtest_bigram_scale = bsxfun(@rdivide, ...
                            Xtest_bigram, scale_bigram_word);

%% test bigram
[test_err_bigram info_bigram ]  = kernel_libsvm(Y,Xtrain_bigram_scale, ...
                                                '-s 4 ',[0.010 ]) ;

%% combine all
% Cross Validation Accuracy = 76.8141% 0.010
% Cross Validation Accuracy = 76.5768% 0.1  Accuracy 99.906
% Cross Validation Accuracy = 67.7733% 0.001
% Cross Validation Accuracy = 76.9926%  (c  = sqrt(2), 0.010)
% Cross Validation Accuracy = 77.1646%
combine_all = [Xtrain_scale (2 .* Xtrain_title_scale) Xtrain_bigram_scale];
[test_err_bigram_title_text info_bigram_title_text] = kernel_libsvm(Y, ...
                                                  combine_all, ['-s ' ...
                    '4'],[0.01],'-v 5');


%% categories predict
categories_train = [train.category];
for i = 1:11
  category_validation(Y,combine_all,categories_train,i);
end


%% predict
combine_test_all = [Xtest_scale (2 .* Xtest_title_scale) Xtest_bigram_scale];
[yhat, acc , vals ] = predict(zeros(size(combine_test_all,1),1), ...
                           combine_test_all, info_bigram_title_text.model,'-b 1') ;

save('-ascii','submit.txt','yhat');

yhat2=load('./submit_c_0.5_stop_word-s-4.txt','-ascii') ;
