function res = calc_rmse(A, B)
% CALC_RMSE - Calculates the RMSE
%   A - predictions
%   B - true labels

res = norm(A - B) / sqrt(size(A, 1));
