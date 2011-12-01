function destroy_knn_model(model)
% DESTROY_MODEL - /pkg/bin/bash
%

for i = 1:model.n_samples
    kdtree_delete(model.tree(i));
end
