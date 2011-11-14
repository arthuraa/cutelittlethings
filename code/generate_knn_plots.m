%% Script/instructions on how to submit plots/answers for question 2.
% Put your textual answers where specified in this script and run it before
% submitting.

% Loading the data: this loads X, Xnoisy, and Y.
load('../data/breast-cancer-data-fixed.mat');

%% 2.1
answers{1} = 'This is where your answer to 2.1 should go. Just as one long string in a cell array';

% Plotting with error bars: first, arrange your data in a matrix as
% follows:
%
%  nfold_errs(i,j) = nfold error with n=j of i'th repeat
%  
% Then we want to plot the mean with error bars of standard deviation as
% folows: y = mean(nfold_errs), e = std(nfold_errs), x = [2 4 8 16].
% 
% >> errorbar(x, y, e);
%
% To add labels to the graph, use xlabel('X axis label') and ylabel
% commands. To add a title, using the title('My title') command.
% See the class Matlab tutorial wiki for more plotting help.
% 
% Once your plot is ready, save your plot to a jpg by selecting the figure
% window and running the command:
%
% >> print -djpg plot_2.1-noisy.jpg % (for noisy version of data)
% >> print -djpg plot_2.1.jpg  % (for regular version of data)
%
% YOU MUST SAVE YOUR PLOTS TO THESE EXACT FILES.

%% 2.2
answers{2} = 'This is where your answer to 2.2 should go. Short and sweet is the key.';

% Save your plots as follows:
%
%  noisy data, k-nn error vs. K --> plot_2.2-k-noisy.jpg
%  noisy data, kernreg error vs. sigma --> plot_2.2-sigma-noisy.jpg
%  regular data, k-nn error vs. K --> plot_2.2-k.jpg
%  regular data, kernreg error vs. sigma --> plot_2.2-sigma.jpg

%% Finishing up - make sure to run this before you submit.
save('problem_2_answers.mat', 'answers');