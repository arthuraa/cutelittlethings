function [nb] = nb_train_pk(X, Y)
% Efficient multi-class naive bayes test code.
%
% Usage:
%
%   [NB] = NB_TRAIN_PK(X, Y)
%
% Trains a Naive Bayes multi-class model. X should be a D x N matrix of N
% sparse examples with D features each. NOTE: THIS IS REVERSED FROM THE WAY WE NORMALLY
% HANDLE THINGS). Y should be a N x K binary indicator matrix indicating which
% class is active at which example.

nb.py = sum(Y)./sum(Y(:));

prior = 1./size(X,2);
k = size(Y,2);
nb.pxy = zeros(size(X,1), k);
for i = 1:k
    yX = bsxfun(@times, X, Y(:,i)');
    nb.pxy(:,i) = [sum(yX, 2) + 1./size(X,2)]./[sum(Y(:,i))+k*prior];
end



