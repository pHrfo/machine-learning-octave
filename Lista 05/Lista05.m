load('ex6data1.mat')

%for i = 1:51,
%  if (y(i) == 1),
%    plot(X(i,1), X(i,2), '.r');
%  else,
%    plot(X(i,1), X(i,2), '.b');
%  endif;
%  hold on;
%endfor;

svmTrain(X,y,1,linearKernel(X(:,1), X(:,2)), 0.0001, 20);