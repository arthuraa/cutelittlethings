function rmse = category_validation_all(Y, X, categories, arg, p)
% CATEGORY_VALIDATION_ALL - Runs category validation with every category.
%

N = max(categories);
rmses = zeros(N, 1);
initial_rmses = zeros(N, 1);

for i = 1:N
    [~, rmses(i), initial_rmses(i)] = ...
        category_validation(Y, X, categories, i, arg, p);
    fprintf('i = %d, RMSE = %f, Initial RMSE = %f\n', ...
            i, rmses(i), initial_rmses(i));
end

% FIXME: maybe this shouldn't be an even mean.
rmse = mean(rmses);
