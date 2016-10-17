imp = importdata("ex1data2.txt");

x = [ones(47,1) imp(:, 1:2)]';
y = imp(:,3)';

epocas = 1000;
alpha = 0.01;
w = [0.0001,0.0001, 0.0001]';

for i = 1:epocas,
  for j = 1:47,
    error_vec(j) = y(j) - w'*(x(:,j)); 
    w = w + alpha*error_vec(j)*x(:, j);
  endfor;
  eqm(i) = error_vec*error_vec'
  
  for j = 1:47,
    pos_aleatoria = floor(mod(1000 * rand(1), 47) + 1);
    
    aux_x = x(:,j);
    aux_y = y(j);
    
    x(:,j) = x(:,pos_aleatoria);
    y(j) = y(pos_aleatoria);
    
    x(:,pos_aleatoria) = aux_x;
    y(pos_aleatoria) = aux_y;
  endfor;
    
endfor;  

eqm = (1/47)*eqm;
subplot(2,1,1);
plot(y);
hold on;
plot (w'*x, 'r');

subplot(2,1,2);
%plot(eqm);

x = x';
y = y';
Wbatch = inv(x'*x)*x'*y;
plot(y);
hold on;
plot(Wbatch'*x','r');