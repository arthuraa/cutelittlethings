%% Example submission: Naive Bayes

%% Load the data
% import test train vocab
format long ;
addpath ./liblinear-1.8/;
addpath ./liblinear-1.8/matlab ;

%% slow loading
load ../data/data_no_bigrams.mat;
load ../data/data_with_bigrams.mat;

%% stem preprocesssing
% stem_file = fopen('stems.txt');
% stems = textscan(stem_file, '%s');
% stems = stems{1};
% fclose(stem_file);
% raw_data = importdata('idf.txt', ' ');
% stem_idx = raw_data.data(:,1) + 1;
% word_weights = raw_data.data(:,2);
% word_weights = word_weights / max(word_weights); % scale word
%                                                  % weights
% t = CTimeleft(numel(train));
% for i = 1:numel(train)
%     t.timeleft();
%     [train_s(i).word_idx, train_s(i).word_count] = ...
%         stem_words(train(i).word_idx, train(i).word_count, ...
%                    stem_idx);
%     [train_s(i).title_idx, train_s(i).title_count] = ...
%         stem_words(train(i).title_idx, train(i).title_count, ...
%                    stem_idx);
% end

%% make feature matrices

Y = double([train.rating])';

% X = make_sparse(train, numel(stems));
% X = X(:, selector);
% Xtrain_title= make_sparse_title(train, numel(stems));
% Xtrain_title = Xtrain_title(:, selector);

% histogram = word_histogram(train, 1, 1, 0, numel(vocab), ...
%                            0);

X = make_sparse(train, numel(vocab));
%X = X(:, histogram.body >= 0);
Xtrain_title = make_sparse_title(train, numel(vocab));
%Xtrain_title = Xtrain_title(:, histogram.title >= 0);
Xtrain_helpful = extract_helpful(train);
Xtrain_helpful_ratio = Xtrain_helpful(:,1) ./ (Xtrain_helpful(:,2) ...
                                               + 0.01);
Xtrain_length = sum(X, 2);


%Xtest = make_sparse(test, size(vocab, 2));
%Xtest_title = make_sparse_title(test, size(vocab, 2));

%% liblinear

%% scale
scale_words = max(max(X), 1); % weird out of memory bug
Xtrain_scale = bsxfun(@rdivide, X', scale_words')';
scale_title = max(max(Xtrain_title), 1);
Xtrain_title_scale = bsxfun(@rdivide, Xtrain_title', ...
                            scale_title')';
scale_helpful = max(max(Xtrain_helpful), 1);
Xtrain_helpful_scale = bsxfun(@rdivide, Xtrain_helpful', ...
                              scale_helpful')';
Xtrain_length_scale = Xtrain_length / max(Xtrain_length);
%Xtrain_scale = scale_features(X, word_weights);
%Xtrain_title_scale = scale_features(Xtrain_title, word_weights);
%Xtest_scale = scale_features(Xtest, word_weights);
%Xtest_title_scale = scale_features(Xtest_title, word_weights);

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

% c = 0.068, p = 9.09, f = 1.149: 0.897321
% c = 0.068, p = 9.09, f = 1.149: 0.895576 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 2.5: 0.894358 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 4: 0.893832 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 8: 0.893862 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 5: 0.893821 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 5.5: 0.893817 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 5.25: 0.893780 (with helpful)
% c = 0.068, p = 9.09, f = 1.149, g = 5.25: 0.892286 (with helpful
% and ratio)
% c = 0.068, p = 9.09, f = 1.149, g = 5.25, h = 5.25: 0.892049 (with helpful
% and ratio)

% 0.891669
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
