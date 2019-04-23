% 2019-04-23 PengXu x1724477385@126.com
% If you want to use the code in your own research, please refer to the papers 
% "Concise Fuzzy System Modeling Integrating Soft Subspace Clustering and Sparse Learning DOI:10.1109/TFUZZ.2019.2895572"
% or "Transfer Representation Learning with TSK Fuzzy System CoRR: abs/1901.02703"
% Test for the one-order TSK-FS

%% Generate randomly constructed data
X_train = rand(100, 10);
Y_train = [ones(50,1);ones(50,1)*2];
X_test = rand(50, 10);
Y_test = [ones(25,1);ones(25,1)*2];

%% Parameter settings
options.omega = 1;
options.k = 5;
options.h = 1;

%% Train and test the model
[train_acc, test_acc] = MyTSK(X_train,Y_train,X_test,Y_test,options);