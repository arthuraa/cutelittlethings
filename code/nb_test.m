function [y] = nb_test(nb, X)
% Generate predictions for a Naive Bayes model with multinomial features.
%
% Usage:
%
%   [Y] = NB_TEST(NB, X)
%
% X is a N x P matrix of N examples with P features each, and NB is a struct
% from the training routine NB_TRAIN. Generates predictions for each of the
% N examples and returns a 0-1 N x 1 vector Y.
% 
% SEE ALSO
%   NB_TRAIN

% Compute log probabilities of P(Y=1)
% YOUR CODE GOES HERE
log_p_y(1) = 
log_p_y(2) = 

log_p_x_and_y = zeros(size(X,1), 2);
for i = 1:size(X,1)
    for y = 1:2
        
        % Compute log P(X,Y) for i'th example:
        % YOUR CODE GOES HERE
        log_p_x_and_y(i,y) = 
    end
end

% Take the maximum of the log generative probability 
[maxp, y] = max(log_p_x_and_y, [], 2);
% Convert from 1,2 based indexing to the 0,1 labels
y = y -1;

    
