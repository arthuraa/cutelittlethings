function results = cross_validation(X, Y, N, f)
    partition = crossvalind('Kfold', size(X, 1), N);
    errors = zeros(N, 1);
    rmses = zeros(N, 1);

    for i = 1:N
        trainingInd = partition ~= i;
        trainingData = X(trainingInd, :);
        trainingLabels = Y(trainingInd, :);
        testData = X(~trainingInd, :);
        testLabels = Y(~trainingInd, :);

        predictions = f(trainingData, trainingLabels, testData);

        errors(i) = mean(testLabels ~= round(predictions));
        rmses(i) = sqrt(mean((testLabels - predictions) .^ 2));
    end

    results.accuracy = mean(errors);
    results.rmse = mean(rmses);
