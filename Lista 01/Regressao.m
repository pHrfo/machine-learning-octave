imp = importdata("ex1data1.txt");

x = imp(1:97, 1);
y = imp(1:97, 2);
X = [ones(97,1) x];

%matx = [ones(97, 1), x];
%inversa = pinv(matx' * matx);
%w = inversa * (matx' * y);


for i = 1:1000,
  soma_erros = 0;
  
  for j = 1:97,
    ycalc = w'*X';
    error_vec(j) = y(j) - ycalc(j);
    w(1) = w(1) + alpha*error_vec(j);
    w(2) = w(2) + alpha*error_vec(j)*x(j);
    soma_erros = soma_erros + error_vec(j)*error_vec(j);
  endfor;
  erro_epoca(i) = soma_erros/(97*2);
  
  pos_aleatoria = randperm(97);
  x = x(pos_aleatoria,:);
  y = y(pos_aleatoria,:);
  X = X(pos_aleatoria,:);
  ycalc = ycalc(:,pos_aleatoria);  
endfor;  

subplot(2,1,1);   
plot(x,y, '.', 'linewidth', 0);
hold on;
plot(x, ycalc,'r','linewidth',1.5);
subplot(2,1,2);
plot(1:1000, erro_epoca);
