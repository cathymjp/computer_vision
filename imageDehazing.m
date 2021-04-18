clc;
clear;
addpath('images');
addpath(genpath('filters'));
addpath(genpath('denoise'));

%% parameters
t0 = 0.01; 
brightest_pixels_ratio = 0.1;
k_size = 9;
omega = 0.95;

%% Step-1: read Image
hazyImage = imread('venus.png'); 
I = cast(hazyImage, 'double'); 
original= imread('Orignalvenus.png');

%% Step-2: Compute atmospheric light (A_c)
A_c  = compute_atmospheric_light(I, k_size, brightest_pixels_ratio);

%% Step-3: Compute Normalized image with A-c
[nI] = compute_nI(I, A_c); 

%% Step-4: Compute Dark Channel
J_dark = compute_dark_channel(nI, k_size); 

%% Step-5: Compute transmission map t(x) 
t_x = compute_tm(I, J_dark, omega); 
t_x1= imguidedfilter(t_x);                  %Fillted Transmission Map

%% Step-6: Recover scene radiance 
J = recover_scene_radiance(I, A_c, t_x, t0); 
J = cast(fix(J), 'uint8');
J1 = recover_scene_radiance(I, A_c, t_x1, t0); 
J1 = cast(fix(J1), 'uint8'); 

%% Bitonic Filter
bitonic_filter = bitonic(t_x, 10); %default
%figure, imshow(bitonic_filter); title('bitonic');
bitonic_filter1 = bitonic(t_x1, 10);
bitonic_cast = recover_scene_radiance(I, A_c, bitonic_filter, t0);
bitonic_cast = cast(fix(bitonic_cast), 'uint8');
%figure, imshow(bitonic_cast); title('bitonic_cast 1');
bitonic_cast1 = recover_scene_radiance(I, A_c, bitonic_filter1, t0);
bitonic_cast1 = cast(fix(bitonic_cast1), 'uint8');


%% Guided image filtering
guided = guidedfilter(t_x, t_x, 4, 5e-2);
guided1 = guidedfilter(t_x1, t_x1, 4, 5e-2);
guided_color_tx1 = guidedfilter_color_sxy(t_x1, t_x1, 4, 5e-2);
guided_color_tx = guidedfilter_color_sxy(t_x, t_x, 4, 5e-2);
guided_cast = recover_scene_radiance(I, A_c, guided, t0); 
guided_cast = cast(fix(guided_cast), 'uint8');
guided_cast1 = recover_scene_radiance(I, A_c, guided1, t0);
guided_cast1 = cast(fix(guided_cast1), 'uint8');
%figure, imshow(guided_color_tx); title('t_x Guided Filter - color default');
% %figure, imshow(guided); title('t_x Guided Filter - default');


%% Weighted Least squares based filtering
wls_filter = wlsFilter(t_x); %default
wls_filter1 = wlsFilter(t_x1);
%wls_filter_omega = wlsFilter(t_x, omega);
%wls_filter_t0 = wlsFilter(t_x, 0.5);
wls_cast = recover_scene_radiance(I, A_c, wls_filter, t0); 
wls_cast = cast(fix(wls_cast), 'uint8');
wls_cast1 = recover_scene_radiance(I, A_c, wls_filter, t0);
wls_cast1 = cast(fix(wls_cast1), 'uint8');
%figure, imshow(wls_filter); title('WLS Filter - default');
%figure, imshow(wls_filter_tx1); title('t_x 1WLS Filter - default');
%figure, imshow(wls_filter_t0); title('WLS Filter - t0');


%% Mutual-structure for joint filtering
eps_I = 1e-4;
eps_G = 5e-4;
r = 3;
lambda_I = 1;
lambda_G = 30;
maxiter = 10;

[I, mutual] = mutual_structure_joint_filtering(I, t_x , r, eps_I, eps_G, lambda_I, lambda_G, maxiter);
[I, mutual1] = mutual_structure_joint_filtering(I, t_x1, r, eps_I, eps_G, lambda_I, lambda_G, maxiter);
mutual_gt = psnr(t_x, mutual);
mutual_gt2 = psnr(t_x1, mutual1);
mutual_cast = recover_scene_radiance(I, A_c, mutual, t0);
mutual_cast = cast(fix(mutual_cast), 'uint8');
mutual_cast1 = recover_scene_radiance(I, A_c, mutual1, t0);
mutual_cast1 = cast(fix(mutual_cast1), 'uint8');

%figure, imshow(G); title('Mutual-Structure for joint filtering');
%figure, imshow(G_1); title('t_x1 Mutual-Structure for joint filtering');

%% Zero-order reverse filtering
funcs = {
    {@(in)(imfilter(t_x, fspecial('gaussian', [7 7], 1))), 'gaussian'}, ...
    {@(in)(bitonic(t_x, 10)), 'Bitonic'}, ... %
    {@(in)(guidedfilter_color_sxy(t_x, t_x, 4, 5e-2)), 'guidedFilterColor'}...
    {@(in)(guidedfilter(t_x, t_x, 4, 5e-2)), 'guidedFilter'}... %
    {@(in)(adaptive_manifold_filter(in, 20, 0.2)), 'adaptiveManifold'}, ...
    {@(in)(wlsFilter(t_x)), 'weightedLeastSquare'}, ...  %
    {@(in)(mutual_structure_joint_filtering(t_x, medfilt2(t_x,[3 3]), r, eps_I, eps_G, lambda_I, lambda_G, maxiter)), 'mutual-structure'}, ...
};

% number of iteration
max_iter = 1;
gt = im2double(bitonic_filter);
gt1 = im2double(bitonic_filter1);
% choose adaptive manifold filter
f_func = funcs{2}{1};
f_name = funcs{2}{2};
input = f_func(gt);  % input = f_func(gt);
input1 = f_func(gt1);

output = input;
output1 = input1;

for k = 1 : max_iter
    foutput = f_func(output);
    foutput1 = f_func(output1);
    % record results
    psnr_gt = psnr(im2double(output), im2double(gt));
    psnr_gt1 = psnr(im2double(output1), im2double(gt1));
    mse_gt = mse(output(:), gt(:));
    mse_gt1 = mse(output1(:), gt1(:));
    mse_dt = mse(foutput(:), input(:));
    mse_dt1 = mse(foutput1(:), input1(:));

    % update 
    output = output + input - f_func(output);
    output1 = output1 + input1 - f_func(output1);
    fprintf('%s Iteration %02d: PSNR - %4.2f\n', f_name, k, psnr(im2double(output), gt));
    fprintf('%s Iteration %02d: PSNR - %4.2f\n', f_name, k, psnr(im2double(output1), gt));
end
%figure, imshow(input); title('filtered input');
%figure, imshow(output); title('reversed result');

zero_order_cast = recover_scene_radiance(I, A_c, output, t0); 
figure, imshow(zero_order_cast); title('recover input');
zero_order_cast = cast(fix(zero_order_cast), 'uint8');
zero_order_cast1 = recover_scene_radiance(I, A_c, output1, t0);
zero_order_cast1 = cast(fix(zero_order_cast1), 'uint8');


%% depth map
d_x = -(1/0.05)*log(double(J));
d_x1 = 1-(1/0.05)*log(double(t_x));
d_x2 = -log(double(t_x1))/(1/50);
d_x3 = -(1/0.01)*log(double(zero_order_cast));
figure, imshow(d_x); title('depth result');
figure, imshow(d_x1); title('depth1 result');
figure, imshow(d_x2); title('depth2 result');
figure, imshow(d_x3); title('depth2 result');

%% Step-7: Display Results
%Bitonic Filter
figure('name','Bitonic Filter results'),
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(bitonic_filter,[]),title('Transmission Map');
subplot(2,3,3), imshow(bitonic_cast,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(bitonic_filter1,[]),title('Filtered TM');
subplot(2,3,6), imshow(bitonic_cast1,[]),title('Dehazed Image');

% Zero-Order Reverse Filtering
figure('name','Zero-Order Reverse Filtering results');
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(output,[]),title('Transmission Map');
subplot(2,3,3), imshow(zero_order_cast,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(output1,[]),title('Filtered TM');
subplot(2,3,6), imshow(zero_order_cast1,[]),title('Dehazed Image'); 

%Guided Image Filtering
figure('name','Guided Image Filtering results'),
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(guided,[]),title('Transmission Map');
subplot(2,3,3), imshow(guided_cast,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(guided1,[]),title('Filtered TM');
subplot(2,3,6), imshow(guided_cast1,[]),title('Dehazed Image');

% WLS Filter
figure('name','Weighted Least Squares based Filtering results'),
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(wls_filter,[]),title('Transmission Map');
subplot(2,3,3), imshow(wls_cast,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(wls_filter1,[]),title('Filtered TM');
subplot(2,3,6), imshow(wls_cast1,[]),title('Dehazed Image');

%Mutual-Structure for Joint Filtering
figure('name','Mutual-Structure for Joint Filtering results'),
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(mutual,[]),title('Transmission Map');
subplot(2,3,3), imshow(mutual_cast,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(mutual1,[]),title('Filtered TM');
subplot(2,3,6), imshow(mutual_cast1,[]),title('Dehazed Image');

%Original / Reference
figure('name','Dehazing results'),
subplot(2,3,1), imshow(hazyImage,[]),title('Hazy Image');
subplot(2,3,2), imshow(t_x,[]),title('Transmission Map');
subplot(2,3,3), imshow(J,[]),title('Dehazed Image');
subplot(2,3,4), imshow(original,[]),title('Orignal Image');
subplot(2,3,5), imshow(t_x1,[]),title('Filtered TM');
subplot(2,3,6), imshow(J1,[]),title('Dehazed Image');

% compute quantitative metrics
C1 = calculate_corr(original,J);
C2 = calculate_corr(original,J1);
PSNR1 = calculate_PSNR(original,J);
PSNR2 = calculate_PSNR(original,J1);

corr_bitonic1 = calculate_corr(original, bitonic_cast);
corr_bitonic2 = calculate_corr(original, bitonic_cast1);
PSNR_bitonic1 = calculate_PSNR(original, bitonic_cast);
PSNR_bitonic2 = calculate_PSNR(original, bitonic_cast1);

corr_zero_order1 = calculate_corr(original, zero_order_cast);
corr_zero_order2 = calculate_corr(original, zero_order_cast1);
PSNR_zero_order1 = calculate_PSNR(original, zero_order_cast);
PSNR_zero_order2 = calculate_PSNR(original, zero_order_cast1);

corr_guided1 = calculate_corr(original, guided_cast);
corr_guided2 = calculate_corr(original, guided_cast1);
PSNR_guided1 = calculate_PSNR(original, guided_cast);
PSNR_guided2 = calculate_PSNR(original, guided_cast1);

corr_wls1 = calculate_corr(original, wls_cast);
corr_wls2 = calculate_corr(original, wls_cast1);
PSNR_wls1 = calculate_PSNR(original, wls_cast);
PSNR_wls2 = calculate_PSNR(original, wls_cast1);

corr_mutual1 = calculate_corr(I, mutual);
corr_mutual2 = calculate_corr(I, mutual1);
PSNR_mutual1 = calculate_PSNR(original, mutual_cast);
PSNR_mutual2 = calculate_PSNR(original, mutual_cast1);

%% function compute_atmospheric_light
function A_c = compute_atmospheric_light(I, k_size, brightest_pixels_ratio)
d = size(I); 
height = d(1);
width = d(2); 
I_dark = compute_dark_channel(I, k_size);

pixels_count = height * width; 
brightest_pixels_count = fix(pixels_count * brightest_pixels_ratio); 
[sorted_pixels, indices] = sort(I_dark(:), 'descend'); 

% Select pixel with the highest intensity 
I_gray = rgb2gray(I); 
max_intensity = -1; 

% assume that all values in I_gray are non-negative 
max_intensity_index = 0; 
for i=1:brightest_pixels_count [y, x] = ind2sub([height, width], indices(i)); 
    current_intensity = I_gray(y, x); 
    if current_intensity > max_intensity max_intensity = current_intensity; 
        max_intensity_index = indices(i); 
    end
end

% Get color of pixel with the highest intensity
[y, x] = ind2sub([height, width], max_intensity_index); 
A_c = squeeze(I(y, x, :)); 
end

%% function to Compute Normalized image with A-c
function nI = compute_nI( I, A_c) 
d = size(I);
A_c_matrix = ones(d, 'double'); 
parfor i=1:3 
    A_c_matrix(:,:,i) = A_c(i); 
end
nI = I ./ A_c_matrix;
end

%% function to compute dark_channel
function I_dark  = compute_dark_channel(I, k_size)
% dark channel
I_c = min(I, [], 3);

% Compute dark channel in the local neighborhood
I_dark = ordfilt2(I_c,1,ones(k_size,k_size));
end

%% function to compute transmission map
function [t_x] = compute_tm(I, Ic_dark , omega ) 
 
% Compute transmission matrix 
t_x = 1 - omega * Ic_dark;
end

%% Function  to restore the image
function  J  = recover_scene_radiance(I, A_c, t_x, t0)
d1 = size(I); 
A = zeros(d1, 'double'); 

for i=1:3 
    A(:,:,i) = A_c(i); 
end

% Compute thresholded variant of transmission 
t_x_thresholded = max(t_x, t0);

% Expand transmission to 3-channel image 
t_x_matrix = zeros(d1, 'double'); 
for i=1:3
    t_x_matrix(:,:,i) = t_x_thresholded;
end

% Do the recovery 
J = (I - A) ./ t_x_matrix + A;
end
