function [p] = dt_value(t, x)
% DT_VALUE - Get the value of the decision tree node for input.
%
% Usage:
%
%   P = DT_VALUE(T, X)
%
% Returns the probability distribution over labels P stored at this node
% given a 1 X D vector input X and tree root T.
%
% SEE ALSO
%    DT_TRAIN, DT_TRAIN_MULTI

node = t; % Start at root
while ~node.terminal
    if x(node.fidx) <= node.fval
        node = node.left;
    else
        node = node.right;
    end
end
p = node.value;
