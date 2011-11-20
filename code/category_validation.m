function [acc rmse initial_rmse] = category_validation(Y,X,categories,i,arg,power)
  if ~exist('i','var')
    i = 1 ;
  end
  if ~exist('arg','var')
    arg = '-s 4 -c 0.010 -q';
  end
  if ~exist('power','var')
    power = 10 ;
  end
  len = size(Y,1);
  index = (categories == i);
  Xtrain = X(~index,:);
  test = X(index,:);

  model = train(Y(~index,:),Xtrain,arg) ;
  [yhat, acc,vals] = predict(Y(index,:),test, model,'-b 1');
  rmse = compute_RMSE(Y(index,:),vals,power);
  initial_rmse = norm(Y(index,:) - yhat ,2) ./ sqrt(size(yhat,1));