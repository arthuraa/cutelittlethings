function nb = nb_train(X, Y, n_vals)
% Train a Naive Bayes model with multinomial features.
%
% Usage:
%
%   [NB] = NB_TRAIN(X, Y, N_VALS)
%
% X is a N x P matrix of N examples with P features each. Y is a N x 1 vector
% of 0-1 class labels. N_VALS is the range of the features in X; 
% each X takes on values in 1...N_VALS. Returns a struct NB with fields:
%    nb.p_y         -- scalar, P(Y=1)
%    nb.p_x_given_y -- P x N_VALS x 2 matrix of conditional probabilities
% 
% SEE ALSO
%   NB_TEST

% Compute p_y, P(Y=1)
% YOUR CODE GOES HERE
nb.p_y = 

% This matrix stores the conditional probabilities: index by 
% (feature, val, class label)
nb.p_x_given_y = zeros(size(X,2), n_vals, 2);
for i = 1:size(X, 2)
    for val = 1:n_vals

        % Compute P(X(i)=val | Y = 0) and P(X(i)=val | Y = 1)
        % YOUR CODE GOES HERE: 
        nb.p_x_given_y(i, val, 1) = 
        nb.p_x_given_y(i, val, 2) = 
    end
end    

% Sanity check: check proper summation to within floating point precision.
approxeq = @(x,y) x <= (y+eps) & x >= (y-eps);
assert(all(all(approxeq(sum(nb.p_x_given_y, 2), 1))), ...
        'Conditional probabilities P(X(i)=1|Y=y), ..., P(X(i)=n_vals|Y=y) must sum to one');