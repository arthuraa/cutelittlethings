function predictions = predict_knn(model, test_data, k)
% PREDICT_KNN -
%

predictions = zeros(size(test_data, 1), model.n_samples);

t = CTimeleft(model.n_samples);
for sample = 1:model.n_samples
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