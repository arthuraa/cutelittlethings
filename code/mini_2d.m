function [i,j,v] = mini_2d(arr)
  [v, row] = min(arr); 
  [c, col ] = min(v);

  i = row(col) ; 
  j = col ; 
  v = arr(i,j);
end 