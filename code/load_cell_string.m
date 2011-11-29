function result = load_cell_string(file_name)
  fid = fopen(file_name); 
  res = textscan(fid,'%s');
  result = res{1};
  fclose(fid);
end 