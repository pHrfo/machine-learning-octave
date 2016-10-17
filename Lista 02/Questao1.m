imp = importdata("ex2data1.txt");

x1 = imp(1:70,1)/100;
x2 = imp(1:70,2)/100;
y  = imp(1:70,3);
X = [ones(70,1) x1 x2];

%PLOTTING DATA GROUPS
%for k = 1:100,
%  if (imp(k,3) == 0),
%    plot(imp(k,1), imp(k,2), '.r');
%    hold on;
%  endif;
%  
%  if (imp(k,3) == 1),
%    plot(imp(k,1), imp(k,2), '.b');
%    hold on;
%  endif;
%endfor;  

%STOCHASTIC DESCENDANT GRADIENT ALGORITHM FOR LOGISTIC REGRESSION
alpha = 0.01;
w = [1,1,1]';

for i = 1:1000,
  soma_erros = 0;
  
  for j = 1:70,
    ycalc(j) = 1/(1 + exp(-w'*X(j,:)'));
    error_vec(j) = y(j) - ycalc(j);
    w = w + alpha*error_vec(j)*X(j,:)';
    soma_erros = soma_erros + error_vec(j)*error_vec(j);
  endfor;
  
  erro_epoca(i) = soma_erros/(70*2);
  
  pos_aleatoria = randperm(70);
  y = y(pos_aleatoria,:);
  X = X(pos_aleatoria,:);
  ycalc = ycalc(:,pos_aleatoria); 
  
endfor;

plot(1:1000, erro_epoca);