imp = importdata("ex2data2.txt");

x1 = imp(1:118,1);
x2 = imp(1:118,2);
y  = imp(1:118,3);
X = mapFeature(x1, x2);
X_plot = [ones(118,1) x1 x2];

%%PLOTTING DATA GROUPS
%for k = 1:118,
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
%
%%STOCHASTIC DESCENDANT GRADIENT ALGORITHM FOR LOGISTIC REGRESSION
alpha = 0.01;
w = ones(496,1)/10;
lambda = 0.25

for i = 1:1000,
  soma_erros = 0;
  
  for j = 1:118,
    ycalc(j) = 1/(1 + exp(-w'*X(j,:)'));
    error_vec(j) = y(j) - ycalc(j);
    if (j==1),
      w = w + alpha*error_vec(j)*X(j,:)';
    else,
      w = w + alpha*(error_vec(j)*X(j,:)' - lambda*w);
    endif;
    soma_erros = soma_erros + error_vec(j)*error_vec(j);
  endfor;
  
  erro_epoca(i) = soma_erros/(118*2);
  
  pos_aleatoria = randperm(118);
  y = y(pos_aleatoria,:);
  X = X(pos_aleatoria,:);
  ycalc = ycalc(:,pos_aleatoria); 
  
endfor;

plotDecisionBoundary(w, X, y);