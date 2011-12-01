function Yhat = predict_aux(X, model, p)
% PREDICT_AUX -
%

[Yhat, acc, vals] = predict(zeros(size(X, 1), 1), ...
                            X, model, '-b 1');
Yhat = prob_prediction(vals, p);
