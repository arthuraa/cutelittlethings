function destroy_knn_model(model)
% DESTROY_MODEL - Free memory
%

for i = 1:model.n_samples
    kdtree_delete(model.tree(i));
end
