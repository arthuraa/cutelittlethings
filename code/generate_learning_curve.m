%% Generates the learning curve for the written portion of the question.
%
% Before running this, you should complete the programming portion of the
% assignment. However, use the "Sanity check" portion of the code to help
% with debugging.

%% Load the data.
load ../data/breast-cancer-data.mat

%% Sanity check your code: train on entire dataset, it should do very well.

nb = nb_train(X, Y, 10);
nb_err = mean(nb_test(nb, X) ~= Y);
fprintf('Sanity checking NB: got %.2f%% train error on entire dataset\n', 100*nb_err);

w = lr_train(X, Y);
lr_err = mean(lr_test(w, X) ~= Y);
fprintf('Sanity checking LR: got %.2f%% train error on entire dataset\n', 100*lr_err);

% These should both get less than 3% error. If they get more, then there is
% something wrong with your training code. :(
if nb_err > 0.03
    disp('Warning! Your NB implementation is most likely wrong.');
end
if lr_err > 0.03
    disp('Warning! Your LR implementation is most likely wrong.');
end

%% Generate the learning curve.
% NOTEL: This could take up to 4-5 minutes to complete (it takes ~30 sec on decent desktop),
% so make sure your code passes the sanity check above first!! 

clear nb_err
clear lr_err

% Train with increasing amounts of data. Repeat many times.
train_pct = [0.1:0.10:0.8];
n_repeat = 20;
for r = 1:n_repeat
    r
    % Get random permutation of the data.
    order = randperm(size(X,1));

    % Split into train/test with increasing amounts of train data, but
    % always with the same test set.
    for i = 1:numel(train_pct)

        test_pct = 1.0 - max(train_pct);
        n_train = ceil(train_pct(i)*size(X,1));
        n_test = floor(test_pct * size(X,1));
        
        train_idx = order(1:n_train);
        test_idx  = order(end-n_test:end);
        
        nb = nb_train(X(train_idx,:), Y(train_idx), 10);
        w  = lr_train(X(train_idx,:), Y(train_idx));
        
        nb_err(r,i) = mean(Y(test_idx)~=nb_test(nb, X(test_idx,:)));
        lr_err(r,i) = mean(Y(test_idx)~=lr_test(w, X(test_idx,:)));
    end
end

%% Plot the results.
errorbar(train_pct.*size(X,1), mean(nb_err), std(nb_err)./sqrt(n_repeat), '-ob');
hold on;
errorbar(train_pct.*size(X,1), mean(lr_err), std(lr_err)./sqrt(n_repeat), '-xr');
hold off;
legend({'Naive Bayes','Logistic Regression'});
xlabel('Training Examples');
ylabel('Test Error');
title('Learning Curves for Naive Bayes and Logistic Regression');
print -djpeg -r72 learning_curves.jpg