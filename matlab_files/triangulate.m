function [ P, error ] = triangulate( M1, p1, M2, p2 )
%% 
% Input:
%       M1 - 3x4 Camera Matrix 1
%       p1 - Nx2 set of points
%       M2 - 3x4 Camera Matrix 2
%       p2 - Nx2 set of points
% Output:
%       P  - 
%       error -

%% Implementation
num_points = size(p1,1);

% Extract the rows
row1 = 1;
row2 = 2;
row3 = 3;

M1_1 = M1(row1, :);
M1_2 = M1(row2, :);
M1_3 = M1(row3, :);

M2_1 = M2(row1, :);
M2_2 = M2(row2, :);
M2_3 = M2(row3, :);

% Extract the points' (x,y) coordinates
x_pos = 1;
y_pos = 2;

p1_x = p1(:,x_pos);
p1_y = p1(:,y_pos);

p2_x = p2(:,x_pos);
p2_y = p2(:,y_pos);

%Create matrix A (AX = 0 where X is the 3d point corresponding to a point
%correspondance)

%format
X = zeros(4,1,num_points);
error = 0;

for i=1:num_points
    A = [p1_x(i,:)*M1_3 - M1_1; p1_y(i,:)*M1_3 - M1_2;...
         p2_x(i,:)*M2_3 - M2_1; p2_y(i,:)*M2_3 - M2_2];
    [~, S, V] = svd(A);
    [~, min1] = min(diag(S));
    X(:,:,i) = V(:,min1);
    X(:,:,i) = normalize_homo(X(:,:,i), 4);
    error = error + ...
        error_calc(M1,M2,X(:,:,i),p1(i,:), p2(i,:));
end

P = X(1:3,:,:);
end

function err = error_calc(M1, M2, P, p1_real, p2_real)
p1_real = p1_real.';
p2_real = p2_real.';
p1_est = M1*P;
p1_est = normalize_homo(p1_est, 3);
p2_est = M2*P;
p2_est = normalize_homo(p2_est, 3);
err1 = pdist2( p1_est(1:2,1).', p1_real(1:2,1).' );
err2 = pdist2( p2_est(1:2,1).', p2_real(1:2,1).' );
err = err1+err2;
end

function normalized = normalize_homo(P, div_pos)
    div = P(div_pos,1);
    div = repmat(div,[div_pos 1]);
    normalized = P./div;
end