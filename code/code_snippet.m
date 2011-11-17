

%% snippet code 

%% Make the testing data 
save Xtest_sparse.mat Xtest

Xtest = make_sparse(test); 
% Make the training data
X = make_sparse(train);
Y = double([train.rating]');



%% feature 
vocab(find(X(5,:))) % get the 5th vocabulary
train(5).text
cell2csv('vocab.txt',vocab,'\n')



%% Naive bayes, provided by David
Yk = bsxfun(@eq, Y, [1 2 4 5]);
nb = nb_train_pk([X]'>0, [Yk]);
Xtest = make_sparse(test, size(X, 2));
Yhat = nb_test_pk(nb, Xtest'>0);
% Convert from classes 1...4 back to the actual ratings of 1, 2, 4, 5
[tmp, Yhat] = max(Yhat, [], 2);
ratings = [1 2 4 5];
Yhat = ratings(Yhat)';
save('-ascii', 'submit.txt', 'Yhat');

