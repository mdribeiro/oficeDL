function W = randInitializeWeights(L_in, L_out,e)

%disp('Test');

epsilon_init = e;
W = zeros(L_out, 1 + L_in);

W = (rand(L_out, 1 + L_in)*2*epsilon_init - epsilon_init);

end
