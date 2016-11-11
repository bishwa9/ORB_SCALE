clear;
clc;

%% read images
im1 = rgb2gray( imread('im1.jpg') );
im2 = rgb2gray( imread('im2.jpg') );

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
%F = estimateFundamentalMatrix(matchedPoints1, matchedPoints2);
H = findHomography(matchedPoints1, matchedPoints2);

%% relative pose
%pose = relativeCameraPose(F); %can't run this on mine? :P