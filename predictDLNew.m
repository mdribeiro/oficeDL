function p = predictDLNew(Theta1, ThetaH, Theta3, X, nhl)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

h1 = ([ones(m, 1) X] * Theta1');

hLast = h1;
hH = zeros(m,size(hLast,2),nhl);
for i=1:nhl

        Theta = ThetaH(:,:,i);
        hH(:,:,i) = ([ones(m, 1) hLast] * Theta');

        hLast = hH(:,:,i);

end

h3 = ([ones(m, 1) hH(:,:,nhl)] * Theta3');
%[dummy, p] = max(h2, [], 2);

p = h3;

% =========================================================================


end
