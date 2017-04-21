function [cost,grad] = sparseAutoencoderCost(theta, visibleSize, hiddenSize, ...
                                             lambda, sparsityParam, beta, data)

% visibleSize: the number of input units (probably 64) 
% hiddenSize: the number of hidden units (probably 25) 
% lambda: weight decay parameter
% sparsityParam: The desired average activation for the hidden units (denoted in the lecture
%                           notes by the greek alphabet rho, which looks like a lower-case "p").
% beta: weight of sparsity penalty term
% data: Our 64x10000 matrix containing the training data.  So, data(:,i) is the i-th training example. 
  
% The input theta is a vector (because minFunc expects the parameters to be a vector). 
% We first convert theta to the (W1, W2, b1, b2) matrix/vector format, so that this 
% follows the notation convention of the lecture notes. 

W1 = reshape(theta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
W2 = reshape(theta(hiddenSize*visibleSize+1:2*hiddenSize*visibleSize), visibleSize, hiddenSize);
b1 = theta(2*hiddenSize*visibleSize+1:2*hiddenSize*visibleSize+hiddenSize);
b2 = theta(2*hiddenSize*visibleSize+hiddenSize+1:end);

% Cost and gradient variables (your code needs to compute these values). 
% Here, we initialize them to zeros. 
cost = 0;
W1grad = zeros(size(W1)); 
W2grad = zeros(size(W2));
b1grad = zeros(size(b1)); 
b2grad = zeros(size(b2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost/optimization objective J_sparse(W,b) for the Sparse Autoencoder,
%                and the corresponding gradients W1grad, W2grad, b1grad, b2grad.
%
% W1grad, W2grad, b1grad and b2grad should be computed using backpropagation.
% Note that W1grad has the same dimensions as W1, b1grad has the same dimensions
% as b1, etc.  Your code should set W1grad to be the partial derivative of J_sparse(W,b) with
% respect to W1.  I.e., W1grad(i,j) should be the partial derivative of J_sparse(W,b) 
% with respect to the input parameter W1(i,j).  Thus, W1grad should be equal to the term 
% [(1/m) \Delta W^{(1)} + \lambda W^{(1)}] in the last block of pseudo-code in Section 2.2 
% of the lecture notes (and similarly for W2grad, b1grad, b2grad).
% 
% Stated differently, if we were using batch gradient descent to optimize the parameters,
% the gradient descent update to W1 would be W1 := W1 - alpha * W1grad, and similarly for W2, b1, b2. 
% 
m = size(data,2);

% phat = zeros(hiddenSize,1);
sp = ones(hiddenSize,1)*sparsityParam;


%forward computing
z2 = W1 * data + repmat(b1,1,m);
a2 = sigmoid(z2);
z3 = W2 * a2 + repmat(b2,1,m);
a3 = sigmoid(z3);

se = trace((data-a3)*(data-a3)')/2;
phat = mean(a2,2);

cost = se/m + lambda * 0.5 * (trace(W1 * W1') + trace(W2 * W2')) + beta*(sparsityParam * sum(log(sp./phat)) + (1-sparsityParam) * sum(log((1-sp)./(1-phat))));
% cost = se/m + lambda*0.5 * (trace(W1 * W1') + trace(W2 * W2'));

% computing error term
% for i = 1:m
%     et3 = -(data(:,i)-a3(:,i)).* fprime(z3(:,i));
%     et2 = W2'*et3 + beta*(-1*sp./phat + (1-sp)./(1-phat));
%     et2 = et2.*(a2(:,i).*(1-z2(:,i)));
%     
%     W1delta = et2 * data(:,i)';
%     W2delta = et3 * a2(:,i)';
%     b1delta = et2;
%     b2delta = et3;
%     
%     W1grad = W1delta + W1grad;
%     W2grad = W2delta + W2grad;
%     b1grad = b1delta + b1grad;
%     b2grad = b2delta + b2grad;
% end
%%
%矢量化操作

et3 = -(data-a3).* fprime(z3);
et2 = W2'* et3 + repmat(beta*(-1*sp./phat + (1-sp)./(1-phat)),1,m);
% et2 = W2'* et3;
et2 = et2.* fprime(z2);

W1grad = et2 * data';
W2grad = et3 * a2';
b1grad = sum(et2,2);
b2grad = sum(et3,2);
%矢量化操作结束

%%
W1grad = W1grad/m + lambda * W1; 
W2grad = W2grad/m + lambda * W2;
b1grad = b1grad/m;
b2grad = b2grad/m;

%-------------------------------------------------------------------
% After computing the cost and gradient, we will convert the gradients back
% to a vector format (suitable for minFunc).  Specifically, we will unroll
% your gradient matrices into a vector.

grad = [W1grad(:) ; W2grad(:) ; b1grad(:) ; b2grad(:)];

end

%-------------------------------------------------------------------
% Here's an implementation of the sigmoid function, which you may find useful
% in your computation of the costs and the gradients.  This inputs a (row or
% column) vector (say (z1, z2, z3)) and returns (f(z1), f(z2), f(z3)). 

function sigm = sigmoid(x)
  
    sigm = 1 ./ (1 + exp(-x));
end

function fp = fprime(z)
    fp = sigmoid(z).* (1 - sigmoid(z));
end