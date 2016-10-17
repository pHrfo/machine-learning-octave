imp =  importdata("ex2data1.txt");

X_train = [ones(80,1) imp(1:80, 1:2)/100];
X_test = [ones(20,1) imp(81:100, 1:2)/100];
y_train = imp(1:80,3);
y_test = imp(81:100,3);

%STOCHASTIC DESCENDANT GRADIENT ALGORITHM FOR LOGISTIC REGRESSION
alpha = 0.01;
w = [1,1,1]';

for i = 1:1000,
  soma_erros = 0;
  
  for j = 1:80,
    ycalc(j) = 1/(1 + exp(-w'*X_train(j,:)'));
    error_vec(j) = y_train(j) - ycalc(j);
    w = w + alpha*error_vec(j)*X_train(j,:)';
    soma_erros = soma_erros + error_vec(j)*error_vec(j);
  endfor;
  
  erro_epoca(i) = soma_erros/(80*2);
  
  pos_aleatoria = randperm(80);
  y_train = y_train(pos_aleatoria,:);
  X_train = X_train(pos_aleatoria,:);
  ycalc = ycalc(:,pos_aleatoria); 
endfor;

for k = 1:20,
  ycalc_test(k) = 1/(1 + exp(-w'*X_test(k,:)'));
  if (ycalc_test(k) <= 0.5),
    ycalc_test(k) = 0;
  else
    ycalc_test(k) = 1;
  endif;
endfor;

subplot(2,1,1);
plot(y_test,".b");
subplot(2,1,2);
plot(ycalc_test,".b");