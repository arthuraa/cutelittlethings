function [fidx val max_ig] = dt_choose_feature_multi(X, Z, Xrange, colidx)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI

% YOUR CODE GOES HERE