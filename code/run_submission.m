

%% Example submission: Naive Bayes


%% add path
format long ; addpath ./liblinear-1.8/; addpath ./liblinear-1.8/matlab ;

%% sparse_mapping
mapping = load('mapping.txt','-ascii');
sparse_mapping = sparse(mapping(:,1),mapping(:,2),1);

%% no_bigrams
load ../data/data_no_bigrams.mat;

%% bigrams
load ../data/data_with_bigrams.mat ;


%% quick loading
load XY_sparse.mat ; load Xtest_sparse ; %% load X Y;
% load 'stop_index.txt' stop_index ; % load the data into stop_index;
% load invalid_index.mat ;


%% liblinear
%% scale
my_data.train = train ;
my_data.test = test ;
clear train ; % conflict with train function ;
clear test ;
my_data.scale_of_x = max(max(X),1); % important, otherwise out of memory


my_data.test_of_x = Xtest(:,1:size(X,2));
my_data.stest_of_x = bsxfun(@rdivide, my_data.test_of_x, my_data.scale_of_x);

Xtrain_title= make_sparse_title(train) ;
my_data.scale_of_title = max(max(Xtrain_title),1);
my_data.strain_of_title = bsxfun(@rdivide, ...
                                 Xtrain_title, my_data.scale_of_title);
my_data.train_of_y = Y ;
Xtest_title = make_sparse_title(test);

my_data.test_of_title = Xtest_title(:,1:size(Xtrain_title,2));
my_data.stest_of_title =  bsxfun(@rdivide, ...
                                 my_data.test_of_title, my_data.scale_of_title);
my_data.train_of_categories = [train.category] ;

Xtrain_helpful = extract_helpful(train);
Xtrain_helpful_ratio = Xtrain_helpful(:,1) ./ (Xtrain_helpful(:,2) ...
                                               + 0.01);
Xtest_helpful = extract_helpful(test) ;
Xtest_helpful_ratio = Xtest_helpful(:,1) ./ (Xtest_helpful(:,2) ...
                                             + 0.01);

scale_helpful = max(max(Xtrain_helpful),1) ;
my_data.scale_of_train_helpful  = scale_helpful ;

Xtrain_helpful_scale = bsxfun(@rdivide, Xtrain_helpful', ...
                              scale_helpful')';

Xtest_helpful_scale = bsxfun(@rdivide, Xtest_helpful' , ...
                             scale_helpful')' ;

my_data.strain_of_helpful = Xtrain_helpful_scale;
my_data.stest_of_helpful = Xtest_helpful_scale ;

Xtest_helpful_scale = bsxfun(@rdivide, Xtest_helpful', ...
                              scale_helpful')';

my_data.stest_helpful = Xtest_helpful_scale ;

my_data.train_helpful_ratio  = Xtrain_helpful_ratio ;
my_data.test_helpful_ratio = Xtest_helpful_ratio ;

Xtrain_length= sum(X,2);
Xtest_length= sum(Xtest,2) ;

scale_length = max(Xtrain_length) ;
Xtrain_length_scale = Xtrain_length ./ scale_length ;
Xtest_length_scale = Xtest_length ./ scale_length ;

my_data.strain_length = Xtrain_length_scale ;
my_data.stest_length = Xtest_length_scale  ;


% need not offset  1
number_junk_words = load('number_junk_words_index.txt','-ascii');
number_junk_words_in_train = number_junk_words(number_junk_words <= size(Xtrain_title,2));
my_data.train_of_x = X ;
my_data.strain_of_x = bsxfun(@rdivide, X, my_data.scale_of_x);
my_data.strain_of_x(:,number_junk_words_in_train) = 0.5 * ...
    my_data.strain_of_x(:,number_junk_words_in_train);

my_data.stest_of_x(:,number_junk_words_in_train) = 0.5 * ...
    my_data.stest_of_x(:, number_junk_words_in_train) ;

%% feature downgrade
% noun_words = load('dict_index.txt','-ascii');
% noun_words = noun_words + 1 ;


% adj_adv_index_in_train = adj_adv_index(adj_adv_index <= ...
%                                                 size(Xtrain_title,2));

% my_data.strain_of_x(:,adj_adv_index_in_train) = ...
%     1 * my_data.strain_of_x(:, ...
%                             adj_adv_index_in_train) ;

%% adj_adv_index
% adj_adv_index = load('adj_adv_index.txt','-ascii');

% adj_adv_featrues = my_data.strain_of_x ;
% adj_adv_featrues(:,~adj_adv_index) = 0 ;

% [test_err info] =  ...
%     kernel_libsvm(my_data.train_of_y, adj_adv_featrues, '-s 4 -q', ...
%               [0.02,0.04, 0.06,0.08], my_data.train_of_categories)

% my_data.strain_of_x(:,noun_words) = 0 ;

% * 2 is not a good idea, it performs pretty bad when doing
% category validation% Xtrain_title_scale Xtrain_title_scale does
% not see a good improvement
%% model
%% parameters c p (-B 1 )
%%
combine_features = [my_data.strain_of_x    ...
                    my_data.strain_of_title  ...
                    5 * my_data.strain_of_helpful ...
                    5 * my_data.train_helpful_ratio ...
                    8 * my_data.strain_length];

%% sweeping
[test_err info] = kernel_libsvm(my_data.train_of_y, combine_features, ...
                                '-s 4 -q ', [ 0.04, 0.05,0.055, 0.06,0.07], ...
                                my_data.train_of_categories);

%% category
for i = 1 :  11
  cats{i} = my_data.train_of_x(my_data.train_of_categories == i,:);
  freq_cat{i} = sum(cats{i},1);
  [sort_freq_cat{i} idx_cat{i} ] = sort(freq_cat{i}, 'descend');
end
%%  category 2 only have 6583 words

freq_table = [freq_cat{1}(:)' ;
         freq_cat{2}(:)' ;
         freq_cat{3}(:)' ;
         freq_cat{4}(:)' ;
         freq_cat{5}(:)' ;
         freq_cat{6}(:)' ;
         freq_cat{7}(:)' ;
         freq_cat{8}(:)' ;
         freq_cat{9}(:)' ;
         freq_cat{10}(:)';
         freq_cat{11}(:)' ;
        ];
%%

% [i,j,val] = find(data)
% data_dump = [i,j,val]
% data = spconvert( data_dump )
% save -ascii data.txt data_dump

%% xx
freq = sum(my_data.train_of_x,1);
sparse_words = vocab(freq == 1);

[sort_freq  idx ] = sort(freq, 'descend') ;
%% I searched bestc bestp (0.08,  9.5 ) -- pretty bad
%% I may tre (0.055, 9.5) later

% Ylabel = my_data.train_of_y ;
% Yfeature = combine_features ;
% model = train(Ylabel , my_data.strain_of_x,  '-s 4  -c 0.08 -q');
% info.model = model ;

%% pred
a = info.vals .^ 9.5 ;
c = bsxfun(@rdivide, a, sum(a,2)) ;
y_test = c * [1;2;4;5]  ;

%% train directly
info.model = train(my_data.train_of_y,combine_features, ...
                '-s  4 -c 0.055 -q ');
%% predict
combine_test_features = [my_data.stest_of_x ...
                         my_data.stest_of_title ...
                         5 * my_data.stest_of_helpful ...
                         5 * my_data.test_helpful_ratio ...
                         8 * my_data.stest_length ]    ;

[yhat_pre,acc_pre ,vals_pre] = predict(zeros(size(combine_test_features),1), ...
                              combine_test_features, info.model, ...
                              '-b 1') ;
%% extract_helpful data
%% p = 10 for 11_16
a = vals_pre.^8.5 ;
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
for i = 1:11
  for p = 10
    [~,rmse,initial_rmse] = category_validation(Y, combine_features, ...
                                   categories_train,i,...
                                                '-s 4 -c 0.08 -q', ...
                                                p);
    fprintf('rmse: %g initial: %g \n',rmse, initial_rmse);
  end
end






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
