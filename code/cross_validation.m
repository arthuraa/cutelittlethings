function [acc rmse] = cross_validation(X, Y, N)
    partition = crossvalind('Kfold', size(X, 1), N);
    errors = zeros(N, 1);
    rmses = zeros(N, 1);
    for i = 1:N
        trainingInd = partition ~= i;
        trainingData = X(trainingInd, :);
        trainingLabels = Y(trainingInd, :);
        testData = X(~trainingInd, :);
        testLabels = Y(~trainingInd, :);

        h = nb_train_pk([trainingData]'>0, [bsxfun(@eq, trainingLabels, [1 2 4 5])]);
        probabilities = nb_test_pk(h, [testData]'>0);
        
        errors(i) =