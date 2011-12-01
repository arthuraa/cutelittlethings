function predictions = predict_least_squares(model, X)
% PREDICT_LEAST_SQUARES -
%
%
%   MODEL - A model learned by TRAIN_LEAST_SQUARES
%   X - Test points

predictions = min(max([X ones(size(X, 1), 1)] * model.alpha, 1), 5);

