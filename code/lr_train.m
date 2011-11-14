function [w] = lr_train(X, Y)
% Train a logistic regression model.
%
% Usage:
%
%   [W] = LR_TRAIN(X, Y)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of 0-1 class labels. Trains a logistic regression model with 1 additional
% bias parameter, which is prepended to the front of the 1 x (P+1) returned 
% weight vector W.
%
% SEE ALSO
%   LR_TEST

% Add a constant feature to each example to learn a bias term.
X = [ones(size(X,1), 1) X];

% Set all weights to 0
w = zeros(1, size(X,2));

% Fix training parameter settings 
step_size = 0.001;
lambda = 1e-3;

% *** NOTE: Change Y from 0,1 to -1, 1 to simplify training expression, and
% match the lecture notes online.
Y(Y==0) = -1;

% Run for at most 1000 iterations: Update weights until they converge
for t = 1:1000
    
    % Compute P(Y|X) for the new weights
    % YOUR CODE GOES HERE
    p_y_given_x = 

    % Compute gradient using P(Y|X)
    % YOUR CODE GOES HERE
    grad = 
    
    % Update weights
    w = w + step_size * (grad - lambda.*w);
    
    % Check for convergence
    if norm(grad) < 5, break;
    end
end

