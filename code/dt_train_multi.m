function root = build_dt(X, Y, depth_limit)
% DT_TRAIN_MULTI - Trains a multi-class decision tree classifier.
%
% Usage:
%
%    tree = dt_train(X, Y, DEPTH_LIMIT)
%
% Returns a tree of maximum depth DEPTH_LIMIT learned using the ID3
% algorithm. Assumes X is a N x D matrix of N examples with D features. Y
% must be a N x 1 {1, ..., K} vector of labels. 
%
% Returns a linked hierarchy of structs with the following fields:
%
%   node.terminal - whether or not this node is a terminal (leaf) node
%   node.fidx, node.fval - the feature based test (is X(fidx) <= fval?)
%                          associated with this node
%   node.value - 1 x K vector of P(Y==K) as predicted by this node
%   node.left - child node struct on left branch (f <= val)
%   node.right - child node struct on right branch (f > val)
%
% SEE ALSO
%    DT_CHOOSE_FEATURE_MULTI, DT_VALUE

% YOUR CODE GOES HERE