function predictions = nb_train_test_max(trainingData, trainingLabels, testData)
    h = nb_train_pk([trainingData]'>0, [bsxfun(@eq, trainingLabels, [1 2 4 5])]);
    probabilities = nb_test_pk(h, [testData]'>0);
    [~, Yhat] = max(probabilities, [], 2);
    ratings = [1 2 4 5];
    predictions = ratings(Yhat)';
