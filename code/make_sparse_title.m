function [X] = make_sparse_title(data, nf)
% Returns a sparse matrix representation of the data.
%
% Usage:
%
%  [X] = MAKE_SPARSE_TITLE(DATA, NF)
%
% Converts the Amazon review data structure into a sparse matrix using the
% first 1....NF word features only. If NF is not specified, then uses all
% features word features encounted in DATA.

colidx = vertcat(data.title_idx);
counts = vertcat(data.title_count);

if nargin==1
    nf = double(max(colidx));
end

keep_idx = find(colidx <= nf);

rowidx = zeros(size(colidx));

idx = 1;

t = CTimeleft(numel(data));
for i = 1:numel(data)
    t.timeleft();    
    for j = 1:numel(data(i).title_idx)
        rowidx(idx) = i;
        idx = idx + 1;
    end
end

X = sparse(rowidx(keep_idx), double(colidx(keep_idx)), double(counts(keep_idx)), numel(data), nf);


