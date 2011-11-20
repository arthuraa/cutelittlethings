function res = test_pd(A)
% TEST_PD -
%
res = all(eig((A + A')/2) > 0);

