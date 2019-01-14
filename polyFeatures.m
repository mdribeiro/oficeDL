%function [X_poly] = polyFeatures(X,p)
function [X_poly] = polyFeatures(X)

%X_poly = zeros(numel(X), p);
X_poly = zeros(length(X), 9);

X_poly(:,1) = X(:,1);
X_poly(:,2) = X(:,2);
X_poly(:,3) = X(:,3);

X_poly(:,4) = X(:,1).*X(:,1);
X_poly(:,5) = X(:,2).*X(:,2);
X_poly(:,6) = X(:,3).*X(:,3);

X_poly(:,7) = X(:,1).*X(:,2);
X_poly(:,8) = X(:,1).*X(:,3);
X_poly(:,9) = X(:,2).*X(:,3);

%for i=2:p
%    X_poly(:,i) = (X(:,1)).^(i);
%end

end
