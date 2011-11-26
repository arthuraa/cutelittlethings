%% Example submission: Naive Bayes

%% Load the data
% import test train vocab
format long ;
addpath ./liblinear-1.8/;
addpath ./liblinear-1.8/matlab ;

%% slow loading
load ../data/data_no_bigrams.mat;
load ../data/data_with_bigrams.mat ;


%% quick loading
load XY_sparse.mat ; %% load X Y;
load Xtest_sparse ;
load 'stop_index.txt' stop_index ; % load the data into stop_index;
load invalid_index.mat ;


%% liblinear

%% test_title

%% scale
scale_word = max(max(X),1); % important, otherwise out of memory
Xtrain_scale = bsxfun(@rdivide, X',scale_word')';
% since we can only use the features we have
Xtest = Xtest(:,1:size(Xtrain_scale,2));
Xtest_scale = bsxfun(@rdivide, Xtest', scale_word')';

Xtrain_title= make_sparse_title(train) ;
scale_title = max(max(Xtrain_title),1);
Xtrain_title_scale = bsxfun(@rdivide, Xtrain_title', scale_title')';
Xtest_title = make_sparse_title(test);
Xtest_title = Xtest_title(:,1:size(Xtrain_title,2)); % since we can only use
Xtest_title_scale = bsxfun(@rdivide, Xtest_title, scale_title);

%% initial version
  % 27787:Cross Validation Accuracy = 70.2872%
  % 27802:Cross Validation Accuracy = 72.3885%
  % 27817:Cross Validation Accuracy = 71.6334%
  % 27832:Cross Validation Accuracy = 70.6138%
  % 27839:Accuracy = 87.3939% (54858/62771)

[test_err_1 info_1 ] = kernel_libsvm(Y,Xtrain_scale,'-s 4 ');
[yhat, ~, vals] = predict(Y,Xtrain_scale,info_1.model,'-b 1');
%% predict for initial
[yhat, acc , vals ] = predict(zeros(size(Xtest_scale,1),1), ...
                           Xtest_scale, info_1.model,'-b 1') ;
%% RMSE
y_exp = vals * [1; 2; 4; 5];
RMSE = norm(Y - y_exp,2)./sqrt(size(Y,1)) ;
RMSE2 = norm(Y-yhat,2) ./ sqrt(size(Y,1));
%% remove stop words by hand
Xtrain_scale_filter = Xtrain_scale ;
Xtrain_scale_filter (:, [59]) = 0 ;
[test_err_stop info_stop ] = kernel_libsvm(Y,Xtrain_scale_filter);
%% try to remove stop words
% our result showed it is decreasing actually -_-
  % 27581:Cross Validation Accuracy = 70.2665%
  % 27596:Cross Validation Accuracy = 72.3567%
  % 27611:Cross Validation Accuracy = 71.5107%
  % 27626:Cross Validation Accuracy = 70.6871%
  % 27633:Accuracy = 87.3636% (54839/62771)

Xtrain_scale_filter = Xtrain_scale ;
Xtrain_scale_filter(:,stop_index)  = 0 ;
Xtest_scale_filter(:,stop_index) = 0 ;
[test_err_2 info_2] = kernel_libsvm(Y,Xtrain_scale_filter) ;


%% try to remove invalid_index
% after remvoing length(invalid_index) = 71103 features, it's decreasing
   % 1067:Cross Validation Accuracy = 67.1648%
   % 1082:Cross Validation Accuracy = 68.5747%
   % 1097:Cross Validation Accuracy = 67.6347%
   % 1112:Cross Validation Accuracy = 66.835%
   % 1119:Accuracy = 75.1732% (47187/62771)
learn_model = info_1.model.w; % learn_model (:,3371) % 'GOOD'
a = sign(learn_model);
step = abs(a(1,:) - a(2,:)) + abs(a(2,:) - a(3,:) ) + abs(a(3,:) - a(4,:)) ;
invalid_index = find(step~=2);
Xtrain_scale_filter = Xtrain_scale ;
Xtrain_scale_filter(:,invalid_index) =  0 ;
Xtest_scale_filter(:,invalid_index) = 0 ;
[test_err info] = kernel_libsvm(Y, Xtrain_scale_filter) ;



%% TODO observe title seperately
  % 88271:Cross Validation Accuracy = 66.4686%
  % 88286:Cross Validation Accuracy = 68.17%
  % 88301:Cross Validation Accuracy = 68.3245%
  % 88316:Cross Validation Accuracy = 67.9342%
  % 88323:Accuracy = 82.2163% (51608/62771)

[test_err_title_1  info_title_1] = kernel_libsvm(Y,Xtrain_title_scale) ;


%% TODO try to combine the title index effectively
  % 88340:Cross Validation Accuracy = 73.5579%
  % 88355:Cross Validation Accuracy = 75.3118%
  % 88370:Cross Validation Accuracy = 73.8016%
  % 88385:Cross Validation Accuracy = 73.0146%
  % 88392:Accuracy = 91.931% (57706/62771)

combine_features = [Xtrain_scale Xtrain_title_scale];
[test_err_combine_1 info_combine_1 ] = kernel_libsvm(Y,combine_features);
[yhat2, ~, vals_2] = predict(Y,combine_features, info_combine_1.model,'-b 1');

%% combine the title index (emphasize the title)
  % 88409:Cross Validation Accuracy = 74.0135%
  % 88424:Cross Validation Accuracy = 75.2163%
  % 88439:Cross Validation Accuracy = 73.3874%
  % 88454:Cross Validation Accuracy = 72.5701%
  % 88461:Accuracy = 92.8645% (58292/62771)
combine_features = [Xtrain_scale Xtrain_title_scale Xtrain_title_scale];
[test_err_combine_1 info_combine_1 ] = kernel_libsvm(Y,combine_features);

%% combine the title index (emphasize the title 2)
% 88795:Cross Validation Accuracy = 73.7857%
% 88810:Cross Validation Accuracy = 73.8956%
% 88825:Cross Validation Accuracy = 72.234%
% 88840:Cross Validation Accuracy = 71.318%
% 88847:Accuracy = 94.1486% (59098/62771)
% sqrt(2) candidate

% * 2 is not a good idea, it performs pretty bad when doing
% category validation
% Xtrain_title_scale Xtrain_title_scale does not see a good improvement
combine_features = [Xtrain_scale    Xtrain_title_scale ];

[test_err_combine_3 info_combine_3 ] = ...
    kernel_libsvm(Y,combine_features, '-s 4', [0.04 0.06 0.08]);

[yhat3, acc, vals_3] = ...
    predict(Y,combine_features, info_combine_3.model, '-b 1');

combine_test_features = [Xtest_scale Xtest_title_scale] ;

[yhat_pre,acc_pre ,vals_pre] = predict(zeros(size(combine_test_features),1), ...
                              combine_test_features, info_combine_3.model, ...
                              '-b 1') ;
%% p = 10 for 11_16
a = vals_pre.^10 ;
c = bsxfun(@rdivide, a, sum(a,2));
y_exp = c * [1;2;4;5];
save('-ascii','submit.txt','y_exp');
%% tweak RMSE
for p = 5:15
  a = vals_3.^p; c = bsxfun(@rdivide,a,sum(a,2));
  y_exp = vals_3 * [1; 2; 4; 5];
  RMSE = norm(Y - y_exp,2)./sqrt(size(Y,1)) ;
  RMSE2 = norm(Y-yhat3,2) ./ sqrt(size(Y,1));
  y_exp2 = c * [1;2;4;5] ;
  RMSE3 = norm(Y - y_exp2,2) ./ sqrt(size(Y,1));
  fprintf('p=%d\n',p);
  disp([RMSE RMSE2 RMSE3])
end
%% using simple_cross_validation to tweak RMSE
for p = 1: 10
  [~,rmse]=simple_cross_validation(Y,combine_features,0.5, ['-s 4 -c ' ...
                      '0.08'], p);
  fprintf('p=%d,RMSEIS %g',p, rmse);
end
%% cross validation
categories_train = [train.category];
combine_features = [Xtrain_scale (1.149*Xtrain_title_scale)];
rmse = ...
    category_validation_all(Y, combine_features, categories_train, ...
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
% [test_err info] = kernel_libsvm(Y,[XScale_filter Xtitle_scale]);

% [test_err info] = kernel_libsvm(Y,[XScale Xtitle_scale]);


% save('-ascii', 'submit.txt', 'yhat');
% %% test
% test_feature_matrix = [Xtest_scale Xtest_title_scale ] ;

% [yhat, ~, vals ] =  predict(zeros(size(test_feature_matrix,1),1), ...
%         test_feature_matrix, info.model) ;

% %% train
% [test_err info ] = kernel_libsvm(Y,XScale);
% [test_err_filter info_filter ] = kernel_libsvm(Y, XScale_filter) ;
% % test_err =   0.108744483917733
% % test_err = 0.12822

% %% predict

% Xtest_scale_filter = Xtest_scale;

% Xtest_scale_filter (:,stop_index) = 0 ;

% [yhat, ~, vals] = predict(zeros(size(Xtest_scale_filter,1),1), ...
%                           Xtest_scale_filter,info_filter.model);

% %% save the file
% save('-ascii', 'submit.txt', 'yhat');

% % learn_model = info_filter.model.w;
% % learn_model (:,3371) % 'GOOD'
% % sum(learn_model) --
% % a = sign(learn_model);
% % step = abs(a(1,:) - a(2,:)) + abs(a(2,:) - a(3,:) ) + abs(a(3,:) - a(4,:)) ;
% % valid_index = find(step==2); % features turn to be 19538;
% % invalid_index = find(step~=2);
% % save invalid_index.mat invalid_index


% %% remove more features
% XScale_filter(:,invalid_index) = 0 ;
% [test_err_filter info_filter ] = kernel_libsvm(Y, XScale_filter) ;

% %% new prediction based on the new model
% Xtest_scale_filter(:,invalid_index) = 0 ;
% [yhat, ~, vals] = predict(zeros(size(Xtest_scale_filter,1),1), ...
%                           Xtest_scale_filter,info_filter.model);
