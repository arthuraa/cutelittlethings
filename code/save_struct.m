%% test file 
fid = fopen('exp.txt', 'w');
fprintf(fid, '%s', 'sucks');
fclose(fid);

%% write product 
fid = fopen('train.product.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.product)) , train);
fclose(fid);

fid = fopen('train.rating.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.rating)) , train);
fclose(fid);

fid = fopen('train.reviewer.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.reviewer)) , train);
fclose(fid);

fid = fopen('train.text.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.text)) , train);
fclose(fid);

fid = fopen('train.title.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.title)) , train);
fclose(fid);


fid = fopen('train.date.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.date)) , train);
fclose(fid);

fid = fopen('train.helpful.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.helpful)) , train);
fclose(fid);
%% numeric value for train data 
fid = fopen('train.category.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%d\n',x.category)) , train);
fclose(fid);

fid = fopen('train.word_idx.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.word_idx))) , train);
fclose(fid);

fid = fopen('train.word_count.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.word_count))) , train);
fclose(fid);

fid = fopen('train.title_idx.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.title_idx))) , train);
fclose(fid);

fid = fopen('train.title_count.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.title_count))) , train);
fclose(fid);




%% write test data 
fid = fopen('test.product.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.product)) , test);
fclose(fid);


fid = fopen('test.reviewer.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.reviewer)) , test);
fclose(fid);

fid = fopen('test.date.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.date)) , test);
fclose(fid);

fid = fopen('test.helpful.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',x.helpful)) , test);
fclose(fid);
%% numeric value for test data 
fid = fopen('test.category.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%d\n',x.category)) , test);
fclose(fid);

fid = fopen('test.word_idx.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.word_idx))) , test);
fclose(fid);

fid = fopen('test.word_count.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.word_count))) , test);
fclose(fid);

fid = fopen('test.title_idx.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.title_idx))) , test);
fclose(fid);

fid = fopen('test.title_count.txt','w' ) ; 
arrayfun( @(x) (fprintf(fid,'%s\n',mat2str(x.title_count))) , test);
fclose(fid);


