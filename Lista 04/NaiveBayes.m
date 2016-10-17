load("ex4data1.data");

#Carregamento dos conjuntos de teste e treinamento
x1 = ex4data1(1:30,1:4);
x2 = ex4data1(51:80,1:4);
x3 = ex4data1(101:130,1:4);

y1 = ex4data1(1:30,5);
y2 = ex4data1(51:80,5);
y3 = ex4data1(101:130,5);

x1_test = ex4data1(31:50,1:4);
x2_test = ex4data1(81:100,1:4);
x3_test = ex4data1(131:150,1:4);

y1_test = ex4data1(31:50,5);
y2_test = ex4data1(81:100,5);
y3_test = ex4data1(131:150,5);

x_train = [x1' x2' x3']';
y_train = [y1' y2' y3']';

x_test = [x1_test' x2_test' x3_test']';
y_test = [y1_test' y2_test' y3_test']';

#Calculo dos parametros do conjunto de treinamento
for i = 1:4,
  #Calculando a media da classe 1
  soma = 0;
  for j = 1:30,
    soma = soma + x1(j,i);
  endfor;
  media(1,i) = soma/30;
  
  #Calculando a media da classe 2
  soma = 0;
  for j = 1:30,
    soma = soma + x2(j,i);
  endfor;
  media(2,i) = soma/30;
  
  #Calculando a media da classe 3
  soma = 0;
  for j = 1:30,
    soma = soma + x3(j,i);
  endfor;
  media(3,i) = soma/30;
  
  #Calculando o desvio padrao da classe 1
  soma = 0;
  for j = 1:30,
    soma = soma + (x1(j,i) - media(1,i))^2;
  endfor;
  desvio(1,i) = sqrt(soma/90);
  
  #Calculando o desvio padrao da classe 2
  soma = 0;
  for j = 1:30,
    soma = soma + (x2(j,i) - media(2,i))^2;
  endfor;
  desvio(2,i) = sqrt(soma/90);
  
  #Calculando o desvio padrao da classe 3
  soma = 0;
  for j = 1:30,
    soma = soma + (x3(j,i) - media(3,i))^2;
  endfor;
  desvio(3,i) = sqrt(soma/90);
endfor;

#Calculando probabilidades de cada exemplo pertencer a cada classe
#Loop dos exemplos
for i = 1:60,
  testVec = x_test(i,:)';
  #CLASSE 1
  sigma = eye(4);
  for j = 1:4,
    sigma(j,j) = desvio(1,j)^2;
  endfor;
  denom = sqrt(det(sigma))*sqrt(2*pi);
  exponential = exp((-1/2)*(testVec - media(1,:)')'*inv(sigma)*(testVec - media(1,:)'));
  probability(i,1) = (1/3)*exponential/denom;
  
  #CLASSE 2
  sigma = eye(4);
  for j = 1:4,
    sigma(j,j) = desvio(2,j)^2;
  endfor;
  denom = sqrt(det(sigma))*sqrt(2*pi);
  exponential = exp((-1/2)*(testVec - media(2,:)')'*inv(sigma)*(testVec - media(2,:)'));
  probability(i,2) = (1/3)*exponential/denom;
  
  #CLASSE 3
  sigma = eye(4);
  for j = 1:4,
    sigma(j,j) = desvio(3,j)^2;
  endfor;
  denom = sqrt(det(sigma))*sqrt(2*pi);
  exponential = exp((-1/2)*(testVec - media(3,:)')'*inv(sigma)*(testVec - media(3,:)'));
  probability(i,3) = (1/3)*exponential/denom;
endfor;

#Classificando os exemplos baseados nas probabilidades calculadas acima
for i = 1:60
  classe = 1;
  comparador = probability(i,classe);
  for c = 2:3,
    if(comparador < probability(i,c)),
      comparador = probability(i,c);
      classe = c;
    endif;
  endfor;
  y_test_results(i) = classe;
endfor;

#Calculando a matriz de confusao
confusion = eye(3);
confusion(1,1) = 20;
confusion(2,2) = 20;
confusion(3,3) = 20;
for i = 1:20,
  if (y_test_results(i) != 1),
    c = y_test_results(i);
    confusion(1,1) = confusion(1,1) - 1;
    confusion(1,c) = confusion(1,c) + 1;
  endif;
endfor;

for i = 21:40,
  if (y_test_results(i) != 2),
    c = y_test_results(i);
    confusion(2,2) = confusion(2,2) - 1;
    confusion(2,c) = confusion(2,c) + 1;
  endif;
endfor;

for i = 41:60,
  if (y_test_results(i) != 3),
    c = y_test_results(i);
    confusion(3,3) = confusion(3,3) - 1;
    confusion(3,c) = confusion(3,c) + 1;
  endif;
endfor;











