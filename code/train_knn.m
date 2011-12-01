function model = train_knn(X, Y, sample_size, n_samples)
% TRAIN_KNN -
%

sample_labels = zeros(sample_size, n_samples);

t = CTimeleft(n_samples);
for i = 1:n_samples
    t.timeleft();
    sample_idx = randperm(size(X, 1));
    sample_idx = sample_idx(1:sample_size);
    sample_data = X(sample_idx, :);
    sample_labels(:, i) = Y(sample_idx);
    tree(i) = kdtree_build(sample_data);
end

model.sample_labels = sample_labels;
model.n_samples = n_samples;
model.tree = tree;
