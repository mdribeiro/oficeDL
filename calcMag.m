function mag = calcMag(grad)

mag = sqrt( grad(:,1).^2 + grad(:,2).^2 + grad(:,3).^2 ) ;

end
