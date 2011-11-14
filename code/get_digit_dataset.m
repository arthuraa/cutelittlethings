function [X Y] = get_digit_dataset(data, numbers, fold)
% GET_DIGIT_DATASET - Get a subset of MNIST digits dataset.
%
% Usage:
%
%   [X Y] = get_digit_dataset(DATA, NUMBERS, FOLD)
%
% Assumes DATA is the MNIST data structure with fields 'train1', 'test1',
% ..., 'train9','test9', representing train/test data for each digit
% respectively. NUMBERS is a cell array containing the digits included in
% the dataset. FOLD is a string, either 'train' or 'test', indicating which
% subset of the data is selected.%
% 
% Example:
% 
%  [X Y] = get_digit_dataset(data, {'7','9'}, 'train');
%
%  Gets a binary (Y in {0,1}) training dataset of 7 v. 9.
%
%  [X Y] = get_digit_dataset(data, {'1','2','3','4'}, 'test');
% 
%  Gets the test set with Y in 1...4 for the digits 1,2,3, and 4.

X = []; Y = [];
for i = 1:numel(numbers)
    input = double(data.([fold numbers{i}]));
    X = [X; input];
    if numel(numbers) == 2
        Y = [Y; repmat(i-1, size(input, 1), 1)];
    else
        Y = [Y; repmat(i, size(input, 1), 1)];
    end
end