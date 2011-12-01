

%% Example submission: Naive Bayes


%% add path
format long ; addpath ./liblinear-1.8/; addpath ./liblinear-1.8/matlab ;

%% sparse_mapping
mapping = load('mapping.txt','-ascii');
sparse_mapping = sparse(mapping(:,1),mapping(:,2),1);
clear mapping ;
%% no_bigrams
load ../data/data_no_bigrams.mat;

%% clear data
my_data.train = train ;
my_data.test = test ;
clear train ; clear test ; % conflict with train function ;

%% bigrams
% load ../data/data_with_bigrams.mat ;


%% quick loading
load XY_sparse.mat ; load Xtest_sparse ; %% load X Y;

X = X * sparse_mapping(1: size(X,2),:) ;
Xtest = Xtest * sparse_mapping ;


% load 'stop_index.txt' stop_index ; % load the data into stop_index;
% load invalid_index.mat ;


%% liblinear
%% scale
my_data.scale_of_x = max(max(X),1); % important, otherwise out of memory
my_data.stest_of_x = bsxfun(@rdivide, my_data.test_of_x, my_data.scale_of_x);

Xtrain_title= make_sparse_title(my_data.train) ;
Xtrain_title = Xtrain_title * sparse_mapping (1: size(Xtrain_title,2),:);
my_data.scale_of_title = max(max(Xtrain_title),1);
my_data.strain_of_title = bsxfun(@rdivide, ...
                                 Xtrain_title, my_data.scale_of_title);
my_data.train_of_y = Y ;

Xtest_title = make_sparse_title(my_data.test);
Xtest_title = Xtest_title * sparse_mapping(1: size(Xtest_title,2),:);

my_data.test_of_title = Xtest_title ;
my_data.stest_of_title =  bsxfun(@rdivide, ...
                                 my_data.test_of_title, my_data.scale_of_title);
my_data.train_of_categories = [my_data.train.category] ;

%% helpful
Xtrain_helpful = extract_helpful(my_data.train);
Xtrain_helpful_ratio = Xtrain_helpful(:,1) ./ (Xtrain_helpful(:,2) ...
                                               + 0.01);
Xtest_helpful = extract_helpful(my_data.test) ;
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

%% finished

% need not offset  1
% number_junk_words = load('number_junk_words_index.txt','-ascii');
% number_junk_words_in_train = number_junk_words(number_junk_words
% <= size(Xtrain_title,2));

my_data.train_of_x = X ;
my_data.strain_of_x = bsxfun(@rdivide, X, my_data.scale_of_x);

% my_data.strain_of_x(:,number_junk_words_in_train) = 0.5 * ...
%     my_data.strain_of_x(:,number_junk_words_in_train);

% my_data.stest_of_x(:,number_junk_words_in_train) = 0.5 * ...
%     my_data.stest_of_x(:, number_junk_words_in_train) ;

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
logic_index_word = sum(my_data.train_of_x,1) > 0;
logic_index_word(1:5773) = 0 ;

logic_index_title = sum(my_data.strain_of_title) > 0 ;
logic_index_title(1:5773) = 0 ;

logic_index = logic_index_word | logic_index_title ;


%% load new_words table
new_words = load_cell_string('mapping_new_words.txt') ;



%% weight correlation for body
cor_body = rating_corr(my_data.strain_of_x(:,logic_index_word), ...
                       my_data.train_of_y);
[vs ,idx ] = sort(abs(cor_body),'descend');

step = find(logic_index_word);
mean_cor_body = mean(abs(cor_body)) ;

cor_body(idx(1:10));
{new_words{step(idx(1:10)) }};



%% weight correlation for title
cor_title = rating_corr(my_data.strain_of_title(:, ...
                                                  logic_index_title), ...
                                     my_data.train_of_y);
[vs_t, idx_t] = sort (abs(cor_title), 'descend');
step_t = find(logic_index_title);
mean_cor_title = mean(abs(cor_title)) ;

%% features
combine_features = [my_data.strain_of_x(:,logic_index_word)    ...
                    my_data.strain_of_title(:,logic_index_title)  ...
                     my_data.strain_of_helpful ...
                     my_data.train_helpful_ratio ...
                     my_data.strain_length];

cor_combine_features = ...
    rating_corr(combine_features,my_data.train_of_y) ;


[combine_res, combine_idx]  = sort(abs(cor_combine_features), ...
                                   'descend');

%% c = 10 15 20 -> better

mean_cor_combine = mean(combine_res) ;
max_cor_combine = max(combine_res);

% scale_c = 0.4 ;

% [test_err info ] = kernel_libsvm(my_data.train_of_y, combine_weighted_features, ...
%                                 '-s 4 -q ', [ 0.05,0.055, 0.06,0.07], ...
%                                 my_data.train_of_categories);

% scale_c = 4 ;

% combine_weighted_features = ...
%     bsxfun(@times, combine_features, ...
%            tan(pi / 4  + scale_c * abs(cor_combine_features)));

scale_c = 20 ;
combine_weighted_features = ...
    bsxfun(@times, combine_features, ...
           exp(scale_c .* (abs (cor_combine_features) - ...
                           mean_cor_combine))) ;

%% 0.05 > 0.06
[test_err info ] = kernel_libsvm(my_data.train_of_y, combine_weighted_features, ...
                                '-s 4 -q ', [ 0.05, 0.06,0.07,0.08], ...
                                my_data.train_of_categories);

%% test features
combine_test_features = [my_data.stest_of_x(:,logic_index_word) ...
                         my_data.stest_of_title(:, logic_index_title) ...
                         my_data.stest_of_helpful ...
                         my_data.test_helpful_ratio ...
                         my_data.stest_length ];

combine_weighted_test_features = ...
    bsxfun(@times, combine_test_features, ...
           exp(scale_c .* (abs(cor_combine_features) - ...
                           mean_cor_combine)));

[yhat_pre,acc_pre ,vals_pre] = predict(zeros(size(combine_test_features),1), ...
                              combine_test_features, info.model, ...
                              '-b 1') ;

%% extract_helpful data
%% p = 10 for 11_16
a = vals_pre.^8.5 ;
c = bsxfun(@rdivide, a, sum(a,2));
y_exp = c * [1;2;4;5];
save('-ascii','submit.txt','y_exp');






% cor_combine_test_features = ...
%     rating_corr(combine_test_features, )

% [combine_test, combine_test_idx]




%% run simple features
simple_features = [my_data.strain_of_x(:,logic_index_word)];
[test_err info] = kernel_libsvm(my_data.train_of_y, simple_features, ...
                                '-s 4 -q ', [ 0.05,0.055, 0.06,0.07], ...
                                my_data.train_of_categories);

%% xx
scale_c  = 10 ;
simple_weighted_features = ...
    bsxfun(@times, simple_features , exp(scale_c .* (abs(cor_body) ...
                                                  - mean_cor_body))) ...
    ;

[test_err info ] = kernel_libsvm(my_data.train_of_y, simple_weighted_features, ...
                                '-s 4 -q ', [ 0.05,0.055, 0.06,0.07], ...
                                my_data.train_of_categories);

%% sweeping
[test_err info] = kernel_libsvm(my_data.train_of_y, combine_features, ...
                                '-s 4 -q ', [ 0.05,0.055, 0.06,0.07], ...
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

[yhat3, acc, vals_3] = predict(Y,combine_features, info_combine_3.model,['-b ' ...
                    '1']);

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
for i = 1:11
  for p = 10
    [~,rmse,initial_rmse] = category_validation(Y, combine_features, ...
                                   categories_train,i,...
                                                '-s 4 -c 0.08 -q', ...
                                                p);
    fprintf('rmse: %g initial: %g \n',rmse, initial_rmse);
  end
end

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
