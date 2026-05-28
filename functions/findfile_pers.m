function data = findfile_pers(filename)

a = dir;
data = [];
for i = 1:length(a)
    if strfind(a(i).name,filename) > 0
        data = a(i).name;
    end
end
