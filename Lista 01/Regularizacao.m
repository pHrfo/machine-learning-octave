imp = importdata("ex1data3.txt");
x_test = [ones(47,1), imp(:,1:5)];
y_test = imp(:,6);

x_train = x_test(1:30,:);
y_train = y_test(1:30,:);

lambda = [0, 1, 2, 3, 4, 5];

I = eye(6);
I(1,1) = 0;
for i=1:6,
  I = eye(6);
  I(1,1) = 0;
    
  w = inv(x_train'*x_train + lambda(i)*I)*x_train'*y_train;
  lambdamat(:,i) = w;   
endfor;