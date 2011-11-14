function [y] = lr_test(w, X)
% Generate predictions for a logistic regression model.
%
% Usage:
%
%   [Y] = LR_TEST(W, X)
%
% X is a N x P matrix of N examples with P features each. W is a 1 x (P+1) 
% vector of weights returned by LR_TRAIN. The output, Y, is a N x 1 vector
% of 0-1 class labels predictions.
%
% SEE ALSO
%   LR_TRAIN

% Add a constant feature to each example for the bias term.
X = [ones(size(X,1), 1) X];

% Compute P(Y=1|X):
% YOUR CODE GOES HERE
p_y = 

% Convert P(Y=1|X) to 0-1 predictions for each example.
y = p_y>=0.5;

