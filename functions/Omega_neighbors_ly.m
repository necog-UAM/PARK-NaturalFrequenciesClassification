function [dim,xx,yy,zz,connmat, dtempl] = Omega_neighbors_ly(source)
 

voxel_inside = find(source.inside==1);
Nvox = length(voxel_inside);


for i=1:3
    pos(:,i) = source.pos(:,i)-min(source.pos(:,i))+1;
    dim(i) = max(pos(:,i));
end

vtempl = zeros(dim);
dtempl = zeros(Nvox,prod(dim));

for v=1:Nvox
    temp = zeros(dim);
    p=pos(voxel_inside(v),:);
    vtempl(p(1),p(2),p(3)) = v;
    temp(p(1),p(2),p(3)) = 1;
    dtempl(v,:) = temp(:);
end

connmat = zeros(Nvox,Nvox);

for v1=1:Nvox
    vpos1 = source.pos(voxel_inside(v1),:);
    for v2=1:Nvox
        vpos2 = source.pos(voxel_inside(v2),:);
        vdif = norm(vpos2-vpos1);
        if vdif<=1.5     % minimal connectivity (7 voxels; 1) / maximal (19; 1.5)
            connmat(v1,v2) = 1;
        end
    end
end
connmat = logical(connmat);

for v=1:Nvox
    ind = find(vtempl==v);
    [i1,i2,i3] = ind2sub(size(vtempl),ind);
    xx(v)=i1;
    yy(v)=i2;
    zz(v)=i3;
end

end