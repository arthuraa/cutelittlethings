function [acc rmse initial_rmse square_error] = category_validation(Y,X,categories,i,arg,power)
  if ~exist('i','var')
    i = 1 ;
  end
  if ~exist('arg','var')
    arg = '-s 4 -c 0.010 -q';
  end
  if ~exist('power','var')
    power = 8 ;
  end
  len = size(Y,1);
  index = (categories == i);
  Xtrain = X(~index,:);
  test = X(index,:);

  model = train(Y(~index,:),Xtrain,arg) ;
  [yhat, acc,vals] = predict(Y(index,:),test, model,'-b 1');

  [square_error, rmse] = compute_RMSE(Y(index,:),vals,power) ;

  % square_error = sum((Y(index, :) - yhat) .^ 2);


  % without using p
  initial_rmse = sqrt(square_error ./ size(yhat,1));
