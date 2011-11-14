
% draw the picture
fplot(@(x)(1 ./ (1 + exp(- x .* [1,2,0.5] ))) , [-14,14] );
legend('1','2','0.5'); % legend the lines 