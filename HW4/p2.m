beta =  [-2; 1; 0]
X = [0 3 1; 1 3 1; 0 1 1; 1 1 1]
y = [1 1 0 0]'
lambda = 0.07;

i = 1;
mu = zeros(4,1);

while(i <= 4)
    mu(i) = 1/(1+exp(-beta'*X(i,:)')) ;
    i = i+1;
end

mu
beta = beta + ((2*lambda*ones(3,3) - X'*diag(mu.*(ones(4,1)-mu))*X)^-1)*(2*lambda*beta - X'*(y-mu))

i=1;
while(i <= 4)
    mu(i) = 1/(1+exp(-beta'*X(i,:)')) ;
    i = i+1;
end

mu
beta = beta + ((2*lambda*ones(3,3) - X'*diag(mu.*(ones(4,1)-mu))*X)^-1)*(2*lambda*beta - X'*(y-mu))

i=1;
while(i <= 4)
    mu(i) = 1/(1+exp(-beta'*X(i,:)')) ;
    i = i+1;
end

mu
