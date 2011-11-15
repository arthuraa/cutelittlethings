function [py ll] = nb_test_pk(nb, X)
% Efficient multi-class naive bayes test code.
%
% Usage:
%
%   [PY LL] = NB_TEST_PK(NB, X)
%
% Where NB is the output of NB_TRAIN_PK and X is a D x N sparse matrix of N
% examples with D features (NOTE: THIS IS REVERSED FROM THE WAY WE NORMALLY
% HANDLE THINGS),  outputs a N x K matrix PY where PY(i,:) is the
% distribution over K classes. LL is the total log likelihood of the data.

ll = 0;
k = numel(nb.py);
py = zeros(size(X,2), k);

for i = 1:k
    logp(i) = log(nb.py(i)) + sum(log(1-nb.pxy(:,i)));
end

t = CTimeleft(size(X, 2));
for i = 1:size(X,2)
    t.timeleft();
    
    xi =  find(X(:,i));
    for j = 1:k
        lp(j) = logp(j) + sum(log(nb.pxy(xi,j))) - sum(log(1-nb.pxy(xi,j)));
    end    

    psum = logsumexp(lp);
    py(i,:) = exp(lp-psum);
    ll = ll + psum;
end

