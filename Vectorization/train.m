
addpath ../sparseAutoencoder/starter/minFunc/
addpath ../sparseAutoencoder/starter/


visibleSize = 28*28;   % number of input units 
hiddenSize = 196;     % number of hidden units 
sparsityParam = 0.1;   % desired average activation of the hidden units.
                     % (This was denoted by the Greek alphabet rho, which looks like a lower-case "p",
		     %  in the lecture notes). 
lambda = 3e-3;     % weight decay parameter       
beta = 3;            % weight of sparsity penalty term  

images = loadMNISTImages('train-images.idx3-ubyte');
labes = loadMNISTLabels('train-labels.idx1-ubyte');

patches = images(:,1:10000);
display_network(patches(:,randi(size(patches,2),200,1)),8);

theta = initializeParameters(hiddenSize, visibleSize);


options.Method = 'lbfgs'; % Here, we use L-BFGS to optimize our cost
                          % function. Generally, for minFunc to work, you
                          % need a function pointer with two outputs: the
                          % function value and the gradient. In our problem,
                          % sparseAutoencoderCost.m satisfies this.
options.maxIter = 400;	  % Maximum number of iterations of L-BFGS to run 
options.display = 'on';


[opttheta, cost] = minFunc( @(p) sparseAutoencoderCost(p, ...
                                   visibleSize, hiddenSize, ...
                                   lambda, sparsityParam, ...
                                   beta, patches), ...
                              theta, options);

                          
                          %% STEP 5: Visualization 

W1 = reshape(opttheta(1:hiddenSize*visibleSize), hiddenSize, visibleSize);
display_network(W1', 12); 

print -djpeg weights.jpg   % save the visualization to a file 