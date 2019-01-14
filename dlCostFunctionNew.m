function [J, grad] = dlCostFunctionNew(nn_params, ...
                                   input, ...
	                           nhl, ...
                                   hidden, ...
                                   result, ...
                                   X, y, lambda)

% Number of elements in the training set data 
m = size(X, 1);
%
% Initializing cost to null
J = 0;

Theta1 = reshape(nn_params(1:hidden * (input + 1)), ...
                 hidden, (input + 1));

Theta1_grad = zeros(size(Theta1));

Delta1 = zeros(size(Theta1));

t1s2 = size(Theta1,2);

arg1 = (hidden * (input + 1));
arg2 = arg1 + (hidden * (hidden + 1));
for i=1:nhl
        ThetaH(:,:,i) = reshape( nn_params((1 + arg1):  (arg2)  ), hidden, (hidden + 1) );
        ThetaH_grad(:,:,i) = zeros(size(ThetaH(:,:,i)));
        tHs2(i) = size(ThetaH(:,:,i),2);
        DeltaH(:,:,i) = zeros(size(ThetaH(:,:,i)));

        arg1 = arg2;
				arg2 = arg2 + (hidden * (hidden + 1));

end

eval(sprintf('Theta%d = reshape(  nn_params((1 + arg1):end), result, (hidden + 1));', nhl + 2));

eval(sprintf('Theta%d_grad = zeros(size(Theta%d));', nhl + 2, nhl + 2));
eval(sprintf('Delta%d = zeros(size(Theta%d));', nhl + 2, nhl + 2));
eval(sprintf('t%ds2 = size(Theta%d,2);', nhl + 2, nhl + 2));

Last = nhl + 2;

%
% % ====================== Forward propagation ======================
%
a1 =  [ones(m, 1) X] ;
a2 = a1 * Theta1' ;
delta2 = zeros(size(a2,2),m);

aLast = a2;
for i=1:nhl

    aH(:,:,i) =  ( [ones(m, 1) aLast ] * transpose(ThetaH(:,:,i)) ) ;
    deltaH(:,:,i) = zeros(size(aH(:,:,i),2),m);
    aLast = aH(:,:,i);

end

aL = aH(:,:,nhl);
eval(sprintf('ThetaLast = Theta%d ;', nhl + 2));
eval(sprintf('a%d = ( [ones(m, 1) aL ] * transpose(ThetaLast ) );', nhl + 3));
eval(sprintf('delta%d = zeros(size(a%d,2),m);', nhl + 3, nhl + 3));

%
% % ====================== Cost function ======================
%

eval(sprintf('aLast = a%d; ', nhl+3));
J = sum((1/(m))*sum( (aLast - y).^2  ) ) ...
+ (lambda/(2*m))*( sum( sum(Theta1(:,2:t1s2).*Theta1(:,2:t1s2) ) ) )...
+ eval(sprintf(' (lambda/(2*m))*( sum( sum(Theta%d(:,2:t%ds2).*Theta%d(:,2:t%ds2) ) ) );',Last,Last,Last,Last));

for i=1:nhl

  J = J + ...
  (lambda/(2*m))*( sum( sum(ThetaH(:,2:tHs2(i),i).* ThetaH(:,2:tHs2(i),i) ) ) );

end

%
% % ====================== Backward propagation ======================
%

eval(sprintf('delta%d(:,1:m) = ( transpose(a%d(1:m,:)) - transpose(y(1:m))  ) ;', Last + 1,Last + 1));

eval(sprintf('deltaLast = delta%d(:,1:m);', Last + 1));
eval(sprintf('thetaLast = Theta%d;', Last ));
for i = nhl:1

  deltaH(:,1:m,i) = ( transpose(thetaLast(:,2:end) ))*(deltaLast(:,1:m)).*calcGradient( transpose([ aH(1:m,:,i) ]) );
  deltaLast = deltaH(:,1:m,i);
  thetaLast = ThetaH(:,:,i);

end

delta2(:,1:m) = (ThetaH(:,2:end,1)')*(deltaH(:,1:m,1)).*calcGradient( [ a2(1:m,:) ]' );

eval(sprintf('Delta%d = Delta%d + delta%d(:,1:m)*( [ ones(m,1) aH(1:m,:,%d) ]  ) ;', Last ,Last, Last + 1, nhl));

for i = nhl:1

  if ( i  == 1 )
    DeltaH(:,:,i) = DeltaH(:,:,i) + deltaH(:,1:m,i)*( [ ones(m,1) a2(1:m,:) ]  );
  else
    DeltaH(:,:,i) = DeltaH(:,:,i) + deltaH(:,1:m,i)*( [ ones(m,1) aH(1:m,:,nhl-1) ]  );
  end


end


Delta1 = Delta1 + delta2(:,1:m)*(a1(1:m,:));

%
Theta1_grad = (1/m)*Delta1 ;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);

eval(sprintf('Theta%d_grad = (1/m)*Delta%d  ;', Last ,Last ));
eval(sprintf('Theta%d_grad(:,2:end) = Theta%d_grad(:,2:end) + (lambda/m)*Theta%d(:,2:end); ;', Last ,Last, Last ));

for i = 1:nhl
  ThetaH_grad(:,:,i) = (1/m)*DeltaH(:,:,i) ;
  ThetaH_grad(:,2:end,i) = ThetaH_grad(:,2:end,i) + (lambda/m)*ThetaH(:,2:end,i);
end

eval(sprintf('Theta_gradLast = Theta%d_grad ;',Last));

% Unroll gradients
grad = [Theta1_grad(:) ; ThetaH_grad(:) ; Theta_gradLast(:)];


end
