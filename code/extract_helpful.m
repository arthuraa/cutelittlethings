function res = extract_helpful(data)
% EXTRACT_HELPFUL -
%

res = zeros(size(data, 2), 2);

for i = 1:size(data, 2)
    if isempty(data(i).helpful)
        res(i, 1) = 0;
        res(i, 2) = 0;
    else
        nums = sscanf(data(i).helpful, '%f of %f');
        res(i, 1) = nums(1);
        res(i, 2) = nums(2);
    end
end



