function predictions = nb_train_test_prob(trainingData, trainingLabels, testData)
    h = nb_train_pk([trainingData]'>0, [bsxfun(@eq, trainingLabels, [1 2 4 5])]);
    probabilities = nb_test_pk(h, [testData]'>0);
    predictions = sum(bsxfun(@times, probabilities, [1 2 4 5]), 2);
