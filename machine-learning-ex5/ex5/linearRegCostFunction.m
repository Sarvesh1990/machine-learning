function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

Cost = 0;
temp = 0;
for i = 1 : m
	XI = X(i , :)';
    YI = y(i, :);		
	temp = (theta' * XI - 	YI)^2;
	Cost = Cost + temp;
end
J = Cost/(2*m);

second_part = 0;
for j = 2 : size(theta,1)
	second_part = second_part + theta(j)^2;
end

J = J + second_part*(lambda/(2*m));

for j = 1 : size(X, 2)
	grad(j) = 0;
	for k = 1 : m 
		grad(j) = grad(j) + (theta' * X(k, :)' - y(k)) * X(k, j);
	end
	if(j > 1)
		grad(j) = grad(j) + (lambda)*theta(j);
	endif
	grad(j) = grad(j)/m;
end	









% =========================================================================

grad = grad(:);

end
