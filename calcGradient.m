function g = calcGradient(z)

g = zeros(size(z));

g = z.*(1 - z);

end
