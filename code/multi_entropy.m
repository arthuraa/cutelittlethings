function [H] = multi_entropy(p)
% MULTI_ENTROPY - Compute H(P(X)) for discrete multi-valued X.
%
% Usage:
% 
%    H = multi_entropy(P)
%
%  Returns the entropy H = -\sum_x p(x) * log(p(x)).
%  For an K X N matrix P, H is a 1 x K vector of entropy for each of the 
%  N distributions over K values.

% YOUR CODE GOES HERE
