function [sq_error rmse] = compute_RMSE(Y,probs,p)
  a = probs.^p;
  c = bsxfun(@rdivide,a,sum(a,2));
  y_exp = c * [1; 2; 4; 5];

  sq_error = sum( (Y-y_exp).^ 2) ; 
  rmse = sqrt (sq_error ./ size (Y,1)) ; 
  % rmse = norm(Y - y_exp,2) ./ sqrt(size(Y,1))