function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%-
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X = [ones(m, 1) X];

hx_1 = sigmoid(Theta1 * X');
hx_1 = hx_1';
hx_1 = [ones(m, 1) hx_1];

hx_2 = sigmoid(Theta2 * hx_1');

for k = 1 : num_labels
	A = (y == k);
	firstPart = -A' * log(hx_2(k, :))';
	secondPart = -(1-A)'*log(1 - hx_2(k, :))';
	J = (J + firstPart + secondPart);
end;

J = J/m;

regularizedValue_1 = 0;

temp = X(:, 2:end);
Theta1_temp = Theta1(:, 2:end);
Theta2_temp = Theta2(:, 2:end);

for j = 1 : size(Theta1_temp, 1)
	for k = 1 : size(temp, 2)
		regularizedValue_1 = regularizedValue_1 + Theta1_temp(j, k)*Theta1_temp(j, k);
	end;
end;

regularizedValue_2 = 0;
for j = 1 : size(Theta2_temp, 1)
	for k = 1 : size(Theta1_temp, 1)
		regularizedValue_2 = regularizedValue_2 + Theta2_temp(j, k)*Theta2_temp(j, k);
	end;
end;


J = J + (lambda/(2*m))*(regularizedValue_1 + regularizedValue_2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

delta_3 = zeros(size(hx_2));


for k = 1 : num_labels 
	A = (y==k);
	delta_3(k, :) = hx_2(k, :) - A';
end;

delta_2 = zeros(size(hx_1));

z2 = Theta1*X';
z2 = [ones(m, 1) z2'];
z2 = z2';

delta_2 = (Theta2'*delta_3).*(sigmoidGradient(z2));
delta_2 = delta_2'(:, 2:end);

Theta1_grad = Theta1_grad + delta_2'*X;
Theta2_grad = Theta2_grad + delta_3*hx_1;

Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

Theta1_reg = Theta1';
Theta1_reg = Theta1_reg(2:end, :);
Theta1_reg = [zeros(size(Theta1_reg, 2), 1) Theta1_reg'	];


Theta2_reg = Theta2';
Theta2_reg = Theta2_reg(2:end, :);
Theta2_reg = [zeros(size(Theta2_reg, 2), 1) Theta2_reg'	];

Theta1_grad = Theta1_grad + (lambda/m).*Theta1_reg;
Theta2_grad = Theta2_grad + (lambda/m).*Theta2_reg;
grad = [Theta1_grad(:) ; Theta2_grad(:)];





end
