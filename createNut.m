B = importdata('dataGrad180All0.csv',delimiterIn,headerlinesIn);
data2 = B.data; clear B;

disp('import data complete');

grad1val = calcMag([ data2(:,8) data2(:,9) data2(:,10)  ]);
grad2val = calcMag([ data2(:,11) data2(:,12) data2(:,13)  ]);
grad3val = calcMag([ data2(:,5) data2(:,6) data2(:,7)  ]);

Xval = [ grad1val  grad2val  grad3val ];

[X_val_poly] = polyFeatures(Xval);

[X_val_norm, mu, sigma] = featureNormalize(X_val_poly);

%clear X;
X2 = X_val_norm;
%X2 = [ ones(size(X2,1),1) X2 ];

y2 = data2(:,2);


nut = predict(Theta1, Theta2, X2);


maxL = max(y2);
minL = min(y2);
% 
% for i=1:length(nut)
% 	if nut(i) > maxL
% 		nut(i) = maxL;
% 	end
% 
% 	if nut(i) < minL
% 		nut(i) = minL;
% 	end
% end
% 
% header = fileread('header');
% bottom = fileread('nut_bottom');
% 
% fid = fopen('nutML','wt');
% fprintf(fid,'%s',header);
% fprintf(fid,'%d\n',size(nut,1));
% fprintf(fid,'%s\n','(');
% for i = 1:size(nut,1)
%     fprintf(fid,'%g\n',nut(i));
% end
% fprintf(fid,'%s',')');
% fprintf(fid,'%s\n',';');
% fprintf(fid,'%s',bottom);
% fclose(fid)

