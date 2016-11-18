clear;
clc;

%% read images
im1 = imread('../test_images/non_planar_imgs/rot1.jpg');
im2 = imread('../test_images/non_planar_imgs/rot2.jpg');

K = [800, 0, 320;
     0, 800, 240;
     0, 0, 1];

%% extract point correspondances corners
im1_corners = detectHarrisFeatures(im1);
im2_corners = detectHarrisFeatures(im2);

[features1,valid_points1] = extractFeatures(im1,im1_corners);
[features2,valid_points2] = extractFeatures(im2,im2_corners);

matches = matchFeatures(features1, features2);

matchedPoints1 = valid_points1(matches(:,1),:);
matchedPoints2 = valid_points2(matches(:,2),:);

figure; showMatchedFeatures(im1,im2,matchedPoints1,matchedPoints2);

%% fundamental matrix
F = estimateFundamentalMatrix(matchedPoints1, matchedPoints2);
%H = findHomography(matchedPoints1, matchedPoints2);

%% essential matrix
E = K' * F * K;
[U,S,V] = svd(E);
m = (S(1,1)+S(2,2))/2;
E = U*[m,0,0;0,m,0;0,0,0]*V';
[U,S,V] = svd(E);

%% relative pose
W = [0, -1, 0;
     1, 0, 0;
     0, 0, 1];

% Make sure we return rotation matrices with det(R) == 1
if (det(U*W*V')<0)
    W = -W;
end

T1 = eye(3,4);
T1s = repmat(T1, [1, 1, 4]);

T2s = zeros(3,4,4);
T2s(:,:,1) = [U*W*V',U(:,3)./max(abs(U(:,3)))];
T2s(:,:,2) = [U*W*V',-U(:,3)./max(abs(U(:,3)))];
T2s(:,:,3) = [U*W'*V',U(:,3)./max(abs(U(:,3)))];
T2s(:,:,4) = [U*W'*V',-U(:,3)./max(abs(U(:,3)))];

%% Visualize the correct one
for i = 1:4
    figure; hold on; axis([-2,2,-2,2,-2,2]);
    plotCamera('Location',T1s(:,4,i)','Orientation',T1s(1:3,1:3,i),'Color',[1,0,0],'Label','1');
    plotCamera('Location',T2s(:,4,i)','Orientation',T2s(1:3,1:3,i),'Color',[0,1,0],'Label','2');
end