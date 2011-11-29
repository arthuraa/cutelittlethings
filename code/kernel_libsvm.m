function [test_err info bestp] = kernel_libsvm(Ylabel,Yfeature, ...
                                          parameter,crange,categories)
  % tuned parameters
    if ~exist('parameter','var')
      model_parameter = '-s 4  ';
    else 
      model_parameter = parameter ;
    end 
    if ~exist('crange', 'var')
      crange = [ 0.02 0.1 0.5 1 ];
    end

    % if ~exist('v','var')
    %   v = '  '
    % end


    



    prange = [8 9 10];
    error = zeros([size(crange,2), size(prange,2)]);
    % error = zeros(size(crange,1),size(prange,1))  ;
    
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % '-s 4 -v 10 -c 100' - too time consuming
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ylabel An n by 1 vector of train labels. (type must be double)
% Yfeature An n by m matrix of n training instances with m features 
%   it must be a sparse matrix. (type must be double)  
  
  % crange = 10.^(-10:2:4); 
  % crange = 10.^(-2:2:2);

  %% for -s 4 c range = 0.5 stands out
  for i = 1:numel(crange)

      % fprintf('-------------------------------------------------\n');
      % par = [model_parameter, ' ',  v , sprintf(' -c %g', crange(i))];
      % fprintf('train parameters: %s\n', par);
      % model = train(Ylabel, Yfeature,par);
   for p = 1:numel(prange)      
     for category = 1:11 
       [~, rmse,~, square_error ] = ... 
           category_validation(Ylabel,Yfeature,categories, category, ... 
                               sprintf('-q -s 4 -c %g ', crange(i)), ...
                               prange(p));
       % fprintf('acc : %g  rmse : % g \n', acc, rmse) ; 
       error(i,p) = error(i,p) + rmse ; 
     end 
     % error(i,p) = sqrt(error(i,p) / size(Ylabel,1));
     error(i,p) = error(i,p) ./ 11 ; 
     fprintf('c : %g p : %g total error rate %g\n',crange(i), ... 
             prange(p) , error(i,p)) ; 
   end 


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % here we have a lot of options to tune 
    % -s 4 multi-class support vector classification
    % -c cost : we sweep it 
    % -e epsilon : set tolerance of termination criterion 
    %     for -s 1,3,4 and 7 Dual maximal violation <= eps;
    %     (default to 0.1)
    % -B bias 
    % -wi weight: weights adjust the parameter C of different classes
    % -v n : n-fold cross validation mode 
    % -q quiet
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % We also implement multi-class SVM by Crammer and Singer (-s 4):
    % min_{w_m, \xi_i}  0.5 \sum_m ||w_m||^2 + C \sum_i \xi_i
    %     s.t.  w^T_{y_i} x_i - w^T_m x_i >= \e^m_i - \xi_i \forall m,i
    % where e^m_i = 0 if y_i  = m,
    %       e^m_i = 1 if y_i != m,
    % Here we solve the dual problem:
    % min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i    %     s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
    % where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    % and C^m_i = C if m  = y_i,
    %     C^m_i = 0 if m != y_i.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L2-regularized logistic regression (-s 0), we solve
    % min_w w^Tw/2 + C \sum log(1 + exp(-y_i w^Tx_i))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L2-regularized L2-loss SVC dual (-s 1), we solve
    % min_alpha  0.5(alpha^T (Q + I/2/C) alpha) - e^T alpha
    %     s.t.   0 <= alpha_i,
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L2-regularized L2-loss SVC (-s 2), we solve
    % min_w w^Tw/2 + C \sum max(0, 1- y_i w^Tx_i)^2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L2-regularized L1-loss SVC dual (-s 3), we solve
    % min_alpha  0.5(alpha^T Q alpha) - e^T alpha
    %     s.t.   0 <= alpha_i <= C,
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L1-regularized L2-loss SVC (-s 5), we solve
    % min_w \sum |w_j| + C \sum max(0, 1- y_i w^Tx_i)^2
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L1-regularized logistic regression (-s 6), we solve
    % min_w \sum |w_j| + C \sum log(1 + exp(-y_i w^Tx_i))
    % where
    % Q is a matrix with Q_ij = y_i y_j x_i^T x_j.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % For L2-regularized logistic regression (-s 7), we solve
    % min_alpha  0.5(alpha^T Q alpha) + \sum alpha_i*log(alpha_i) + \sum (C-alpha_i)*log(C-alpha_i) - a constant
    %     s.t.   0 <= alpha_i <= C,
    % If bias >= 0, w becomes [w; w_{n+1}] and x becomes [x; bias].
    % The primal-dual relationship implies that -s 1 and -s 2 give the same
    % model, and -s 0 and -s 7 give the same.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % We implement 1-vs-the rest multi-class strategy. In training i
    % vs. non_i, their C parameters are (weight from -wi)*C and C,
    % respectively. If there are only two classes, we train only one
    % model. Thus weight1*C vs. weight2*C is used. See examples below.
    % We also implement multi-class SVM by Crammer and Singer (-s 4):
    % min_{w_m, \xi_i}  0.5 \sum_m ||w_m||^2 + C \sum_i \xi_i
    %     s.t.  w^T_{y_i} x_i - w^T_m x_i >= \e^m_i - \xi_i \forall m,i
    % where e^m_i = 0 if y_i  = m,
    %       e^m_i = 1 if y_i != m,
    % Here we solve the dual problem:
    % min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i    %     s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i
    % where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    % and C^m_i = C if m  = y_i,
    %     C^m_i = 0 if m != y_i.
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  end 
  [bestc,bestp, error ] =  mini_2d(error) ; 

  fprintf('Cross-val-liblinear chose best C = %g best p = % g, error : %g \n', ... 
          crange(bestc), ...
          prange(bestp), ... 
          error); 
  
  model = train(Ylabel,Yfeature, ...
                [model_parameter, sprintf(' -c %g', crange(bestc))]);
  
  [yhat, ~, vals] = predict(Ylabel,Yfeature,model,'-b 1');
  % predict(testing_label, testing_feature,model, options)
  % testing_label : if unknown, just give random values 
  % testing_feature: an n by m matrix of n testing instances with m
  %    features 
  % options :
  %   -b : probability_estimates : whether or not (default to 0)
  %        for logistic regression only
  % for output :
  % [predict_label, accuracy, decision_values/prob_estimates] 
  info.vals = vals ; 
  info.yhat = yhat;
  info.model = model ; 
  test_err = mean(yhat ~=Ylabel);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  % Q: LIBLINEAR is slow for my data (reaching the maximal number of iterations)? % Very likely you use a large C or don't scale data. If your number of features is small, you may use the option
  % -s 2
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
  
    
  
  