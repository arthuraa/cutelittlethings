function predictions = predict_knn(model, test_data, k)
% PREDICT_KNN - Uses KD trees learned with train_knn to predict
% labels.
%   MODEL - model learned by train_knn
%   TEST_DATA - data to predict labels on
%   K - how many neighbors

predictions = zeros(size(test_data, 1), model.n_samples);

t = CTimeleft(model.n_samples);
for sample = 1:model.n_samples
    % Running knn on the entire training set is too expensive, so
    % we take several samples from the training set.
    t.timeleft();
    labels = model.sample_labels(:, sample);
    if k == 1
        neighbors = kdtree_nearest_neighbor(model.tree(sample), test_data);
        predictions(:, sample) = labels(neighbors);
    else
        for i = 1:size(test_data, 1)
            neighbors = kdtree_k_nearest_neighbors(model.tree(sample), ...
                                                   test_data(i, :), k);
            predictions(i, sample) = mean(labels(neighbors));
        end
    end
end

predictions = mean(predictions, 2);