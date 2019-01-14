%% Machine Learning OFICE
%

%% Initialization
clear ; close all; clc

%% Import data
headerlinesIn = 1;
delimiterIn = ',';
%A = importdata('dataSet-1800.csv',delimiterIn,headerlinesIn);
%A = importdata('dataGrad180Train0.csv',delimiterIn,headerlinesIn);
A = importdata('data180cyc12All0.csv',delimiterIn,headerlinesIn);
data = A.data; clear A;
data = data(randperm(length(data)),:);

disp('Import complete');

%grad1 = calcMag([ data(:,4) data(:,5) data(:,6)  ]);
%grad2 = calcMag([ data(:,7) data(:,8) data(:,9)  ]);
%grad3 = calcMag([ data(:,10) data(:,11) data(:,12)  ]);

%grad1 = calcMag([ data(:,10) data(:,11) data(:,12)  ]);
%grad2 = calcMag([ data(:,16) data(:,17) data(:,18)  ]);
%grad3 = calcMag([ data(:,7) data(:,8) data(:,9)  ]);


%X = [ grad1 grad2 grad3 ];

X = [ data(:,9) data(:,10) data(:,11) ];


%[X_poly] = polyFeatures(X);

[X_poly] = X;

[X_norm, mu, sigma] = featureNormalize(X_poly);

clear X;
X = X_norm;

%y = data(:,13);


y = data(:,2);


%% Initialize NN parameters

input  = size(X_norm,2);  % 20x20 Input Images of Digits

neurons = 10

nhl = 3
hidden = neurons;

result = 1;          % 10 labels, from 1 to 10

e = 0.05;
initial_Theta1 = randInitializeWeights(input, hidden,e);

initial_TH = zeros(hidden, hidden + 1, nhl);
%strcat('initial_Theta', num2str(i))

for i=1:nhl
	initial_TH(:,:,i) = randInitializeWeights(hidden, hidden,e);
end

eval(sprintf('initial_Theta%d = randInitializeWeights(hidden, result,e);', nhl + 2));

eval(sprintf('initial_Last = initial_Theta%d ;',nhl + 2));

%initial_nn_params = [initial_Theta1(:) ; initial_TH(:) ; initial_Theta3(:)];
initial_nn_params = [initial_Theta1(:) ; initial_TH(:) ; initial_Last(:)];

%% Train NN

options = optimset('MaxIter', 100);

 %  You should also try different values of lambda
 lambda = 0

 % Create "short hand" for the cost function to be minimized
 costFunction = @(p) dlCostFunctionNew(p, ...
                                    input, ...
				    nhl, ...
                                    hidden, ...
                                    result, X, y, lambda);

 [nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

%% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden * (input + 1)), ...
                 hidden, (input + 1));

arg1 = (hidden * (input + 1));
arg2 = arg1 + (hidden * (hidden + 1));
for i=1:nhl
        ThetaH(:,:,i) = reshape( nn_params((1 + arg1):  (arg2)  ), hidden, (hidden + 1) );
				arg1 = arg2 ;
				arg2 = arg2 + (hidden * (hidden + 1));

end

%eval(sprintf('Theta%d = reshape(nn_params((1 + (hidden * (input + 1)) + (hidden * (hidden + 1))):end), result, (hidden + 1));', nhl + 2));
eval(sprintf('Theta%d = reshape(   nn_params((1 + arg1):end), result, (hidden + 1)  );', nhl + 2)  );

eval(sprintf('Theta_Last = Theta%d ;',nhl + 2));



%% Predict nut
nut = predictDLNew(Theta1, ThetaH, Theta_Last, X, nhl);

plot(sort(nut)), hold on, plot(sort(y))  ;
