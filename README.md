
# Simplex

```matlab
clc;
clear;

%% --------------------------------------------------
%% QUESTION:
% Maximize Z = x1 + 2x2 + 3x3
%
% Subject to:
%   x1 + 2x2 <= 20
%   3x1 + 4x3 <= 30
%   x1, x2, x3 >= 0
%% --------------------------------------------------

%% STEP 1: Define matrices
C = [1 2 3 0 0];   % Objective coefficients

A = [1 2 0 1 0;
     3 0 4 0 1];

b = [20; 30];

[m,n] = size(A);

%% STEP 2: Initial tableau
T = [A b];

bv = n-m+1:n;   % Basic variables (slack variables)

%% STEP 3: Simplex iterations
while true
    
    % Compute Zj - Cj
    ZjCj = C(bv) * T(:,1:n) - C;
    
    % Check optimality
    if all(ZjCj >= 0)
        break;
    end
    
    % Entering variable (most negative)
    [~, ev] = min(ZjCj);
    
    % Ratio test
    ratio = inf(m,1);
    for i = 1:m
        if T(i,ev) > 0
            ratio(i) = T(i,end) / T(i,ev);
        end
    end
    
    % Leaving variable
    [~, lv] = min(ratio);
    
    % Update basis
    bv(lv) = ev;
    
    % Pivot operation
    pivot = T(lv,ev);
    T(lv,:) = T(lv,:) / pivot;
    
    for i = 1:m
        if i ~= lv
            T(i,:) = T(i,:) - T(i,ev) * T(lv,:);
        end
    end
end

%% STEP 4: Final solution
sol = zeros(1,n);
sol(bv) = T(:,end);

disp('Optimal Solution:')
disp(sol)

disp('Maximum Z:')
disp(C * sol')
```

# BIGM

```matlab
clc;
clear;

%% --------------------------------------------------
%% QUESTION:
% Maximize Z = 3x1 + 2x2
%
% Subject to:
%   x1 + x2 >= 4
%   2x1 + x2 = 5
%   x1, x2 >= 0
%
% Solve using Big M Method
%% --------------------------------------------------

%% STEP 1: Convert to standard form
% x1 + x2 - s1 + a1 = 4
% 2x1 + x2      + a2 = 5

M = 1000;   % very large number

C = [3 2 0 -M -M];   % objective (penalty for artificial variables)

A = [1 1 -1 1 0;
     2 1  0 0 1];

b = [4; 5];

%% STEP 2: Create initial tableau
[m,n] = size(A);
T = [A b];

bv = [4 5];   % artificial variables are initially basic

%% STEP 3: Iterations (same as simplex)
for iter = 1:20
    
    % Current basic solution
    Xb = T(:,end);
    Cb = C(bv);
    
    % STEP 4: Compute Zj - Cj
    ZjCj = Cb * T(:,1:n) - C;
    
    disp('Table:')
    disp([ZjCj 0; T])
    
    % STEP 5: Check optimality
    if all(ZjCj >= 0)
        break;
    end
    
    % STEP 6: Entering variable (most negative)
    [~, ev] = min(ZjCj);
    
    % STEP 7: Ratio test (leaving variable)
    ratio = inf(m,1);
    for i = 1:m
        if T(i,ev) > 0
            ratio(i) = Xb(i)/T(i,ev);
        end
    end
    
    [~, lv] = min(ratio);
    
    % Update basis
    bv(lv) = ev;
    
    % STEP 8: Pivot operation
    pivot = T(lv,ev);
    T(lv,:) = T(lv,:) / pivot;
    
    for i = 1:m
        if i ~= lv
            T(i,:) = T(i,:) - T(i,ev)*T(lv,:);
        end
    end
end

%% STEP 9: Final solution
sol = zeros(1,n);
sol(bv) = T(:,end);

disp('Optimal Solution:')
disp(sol)

disp('Maximum Z:')
disp(C * sol')
```

# Least Cost

```matlab
clc;
clear;

%% --------------------------------------------------
%% QUESTION:
% Solve the following Transportation Problem using Least Cost Method
%
% Cost Matrix:
%     11   13   17   14
%     16   18   14   10
%     21   24   13   10
%
% Supply  = [10 5 9]
% Demand  = [8 7 15 4]
%% --------------------------------------------------

%% STEP 1: Input
cost = [11 13 17 14;
        16 18 14 10;
        21 24 13 10];

supply = [10 5 9];
demand = [8 7 15 4];

%% STEP 2: Balance the problem (if needed)
if sum(supply) < sum(demand)
    cost(end+1,:) = 0;
    supply(end+1) = sum(demand) - sum(supply);
elseif sum(supply) > sum(demand)
    cost(:,end+1) = 0;
    demand(end+1) = sum(supply) - sum(demand);
end

[m,n] = size(cost);

X = zeros(m,n);       % allocation matrix
orig_cost = cost;     % save original cost for final calculation

%% STEP 3: Allocation (Least Cost Rule)
while any(supply > 0) && any(demand > 0)
    
    % Find minimum cost cell
    min_cost = min(cost(:));
    [r,c] = find(cost == min_cost,1);
    
    % Allocate minimum of supply and demand
    alloc = min(supply(r), demand(c));
    
    X(r,c) = alloc;
    
    % Update supply & demand
    supply(r) = supply(r) - alloc;
    demand(c) = demand(c) - alloc;
    
    % Mark this cell as used
    cost(r,c) = inf;
end

%% STEP 4: Display Results
disp('Allocation Matrix:')
disp(X)

total_cost = sum(sum(X .* orig_cost));

disp('Total Cost:')
disp(total_cost)
```

# Steepest Descent

```matlab
clc;
clear;

%% --------------------------------------------------
%% QUESTION:
% Minimize f(x1, x2) = x1^2 - x1*x2 + x2^2
%
% Initial point: x0 = (1 , 0.5)
% Use Steepest Descent Method
%% --------------------------------------------------

%% STEP 1: Define function manually
f = @(x1,x2) x1^2 - x1*x2 + x2^2;

%% STEP 2: Define gradient manually
% df/dx1 = 2x1 - x2
% df/dx2 = -x1 + 2x2

grad = @(x) [2*x(1) - x(2);
             -x(1) + 2*x(2)];

%% STEP 3: Define Hessian manually
H = [2 -1;
     -1 2];

%% STEP 4: Initial point
x = [1; 0.5];

tol = 0.01;
max_iter = 10;

%% STEP 5: Iterations
for i = 1:max_iter
    
    g = grad(x);   % gradient
    
    if norm(g) < tol
        break;
    end
    
    S = -g;   % descent direction
    
    % step size
    alpha = (S'*S)/(S'*H*S);
    
    % update
    x = x + alpha*S;
end

%% STEP 6: Output
disp('Optimal Point:')
disp(x)

disp('Minimum Value:')
disp(f(x(1),x(2)))
```


