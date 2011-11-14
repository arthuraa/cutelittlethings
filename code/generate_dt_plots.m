%% Script/instructions on how to submit plots/answers for question 3.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data
data = load('../data/mnist_all.mat');

% Running a training set for binary decision tree classifier
[X Y] = get_digit_dataset(data, {'7','9'}, 'train');

% Train a depth 4 binary decision tree
dt = dt_train(X, Y, 4);

%% 3.1
answers{1} = 'This is where your answer to 3.1 should go. Just as one long string in a cell array';

% Saving your plot: once you have succesfully plotted your data; e.g.,
% something like:
% >> plot(depth, [train_err test_err]);
% Remember: You can save your figure to a .jpg file as follows:
% >> print -djpg plot_3.1.jpg

%% 3.2
answers{2} = 'This is where your answer to 3.2 should go. Short and sweet is the key.';

% Saving your plot: once you've computed M, plot M with the plotnumeric.m
% command we've provided. e.g:
% >> plotnumeric(M);
%
% Save your file to plot_3.2.jpg
%
% ***** ALSO *******
% Save your confusion matrix M to a .txt file as follows:
% >> save -asci confusion.txt M

%% 3.3
answers{3} = 'This is where your answer to 3.3 should go. Please be concise.';

% E.g., if Xtest(i,:) is an example your method fails on, call:
% >> plot_dt_digit(tree, Xtest(i,:));
%
% Save your file to plot_3.3.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_3_answers.mat', 'answers');