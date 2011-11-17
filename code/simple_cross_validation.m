function [acc rmse] = simple_cross_validation(Y,X,p,arg,power)
  if ~exist('arg','var')
    arg = '-s 4 -c 0.010 ';
  end
  if ~exist('power','var')
    power = 10 ; 
  end 
  len = size(Y,1); 
  index = rand(len,1)<p; 
  Xtrain= X(index,:);
  test= X(~index,:); 
  
  model = train(Y(index,:),Xtrain,arg) ; 
  [yhat, acc,vals] = predict(Y(~index,:),test, model,'-b 1'); 

  rmse = compute_RMSE(Y(~index,:),vals,power) ; 
  % fprintf('RMSE: %d\n', rmse);