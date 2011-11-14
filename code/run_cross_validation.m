function results = run_cross_validation(train)
    X = make_sparse(train);
    Y = double([train.rating]');

    results.naive_bayes = cross_validation(X, Y, 4, @nb_train_test);


