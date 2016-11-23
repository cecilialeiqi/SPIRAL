function runme(filename)
file_dir='../UCR_TS_Archive_2015/';

Train=load(strcat(file_dir,filename,'/',filename,'_TRAIN'));
Test=load(strcat(file_dir,filename,'/',filename,'_TEST'));

label_train=Train(:,1);
Train=Train(:,2:size(Train,2));
label_test=Test(:,1);
Test=Test(:,2:size(Test,2));

X={};
n=size(Train,1);
for i=1:n
	X{i}=Train(i,:)';
end

for i=n+1:n+size(Test,1)
	X{i}=Test(i-n,:)';
end
n=size(X,2)
m=n*20*ceil(log(n));
if (2*m>n*n)
	m=floor(n*n/2)
end
[D,Omega,d]=construct_sparse(X,n,m);
X0=zeros(n,30);
options.maxiter=20;
tic;X_train=matrix_completion_sparse_mex(D,d,Omega,X0,options);toc
Train=[label_train,X_train(1:size(Train,1),:)];
Test=[label_test,X_train(size(Train,1)+1:size(X_train,1),:)];
csvwrite(strcat(file_dir,filename,'/',filename,'_sparse_Train'),Train);
csvwrite(strcat(file_dir,filename,'/',filename,'_sparse_Test'),Test);
%save features for Train/Test data

end

