function [error_train, error_val ] = learningCurveNew(X,y,Xval,yval,lambda,nhl,neurons,maxIt)

m = size(X,1);

error_train = zeros(m,1);
error_val = zeros(m,1);

options = optimset('MaxIter', maxIt);

hidden = neurons;
result = 1;          % 10 labels, from 1 to 10
e = 0.05;


%  You should also try different values of lambda
lambda = 0;

horAxis = 1:100:10000;

cont = 2;

%for i=2:100  %1000
for i=2:100:10000

	disp(['Training example = ', num2str(i)]);

        input  = size(X,2); 

	initial_Theta1 = randInitializeWeights(input, hidden,e);

	initial_TH = zeros(hidden, hidden + 1, nhl);

	for j=1:nhl
        	initial_TH(:,:,j) = randInitializeWeights(hidden, hidden,e);
	end

	eval(sprintf('initial_Theta%d = randInitializeWeights(hidden, result,e);', nhl + 2));

	eval(sprintf('initial_Last = initial_Theta%d ;',nhl + 2));

	initial_nn_params = [initial_Theta1(:) ; initial_TH(:) ; initial_Last(:)];

	%% Train NN
	% Create "short hand" for the cost function to be minimized

	costFunction = @(p) dlCostFunctionNew(p, ...
        	                            input, ...
                	                    nhl, ...
                        	            hidden, ...
                                	    result, X, y, lambda);

 	[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%        error_train(i) = dlCostFunctionNew(nn_params, input, nhl, hidden, result ,X(1:i,:), y(1:i), lambda);
%        error_val(i) = dlCostFunctionNew(nn_params, input, nhl, hidden, result ,Xval, yval, lambda);

 	error_train(cont) = dlCostFunctionNew(nn_params, input, nhl, hidden, result ,X(1:i,:), y(1:i), lambda);
	error_val(cont) = dlCostFunctionNew(nn_params, input, nhl, hidden, result ,Xval, yval, lambda);


	maxV = 1*10^-9;
	minV = 1*10^-12;  %-1e-9
	%plot([1:i],error_train(1:i),'-b',[1:i],error_val(1:i),'-r',[1,i],[1e-11,1e-11],'-k'); ylim([minV maxV]);
	plot(horAxis(1:cont),error_train(1:cont),'-b',horAxis(1:cont),error_val(1:cont),'-r'); ylim([minV maxV]);
	xlabel('Training examples');
	ylabel('Error');
	legend({'Training Set','Validation Set','Reference Error = 1e-11'},'Location','northwest');
 	drawnow;
	cont = cont + 1;
   
end
