% Identify several different types of small, readily available objects, and
% gather several instances of each.

clc;
clear all;
close all;

addpath("images");

image = imread("coin.jpg");
I = rgb2gray(image);

%% <Threshold and remove noise>
level = graythresh(I);
BW = imbinarize(I, level);   % Threshold
BW2 = bwareaopen(BW, 120);    % Remove noise
% imshow(BW);
% imshow(BW2);

figure('Name','Threshold'), 
subplot(2,2,1), imshow(image,[]),title('Input Image');
subplot(2,2,2), imshow(imbinarize(I,'adaptive'),[]), title('Adaptive Threshold');
subplot(2,2,3), imshow(BW, []), title('Otsu Threshold');
subplot(2,2,4), imshow(BW2, []), title('Noise Removed');


%% <Compute various properties of the foreground regions>
%% Structure Elements
SE1 = strel(3);
SE2 = strel('arbitrary',3);
SE3 = strel('diamond',5);
SE4 = strel('disk',15);
SE5 = strel('octagon',30);
SE6 = strel('line',10,45);
SE7 = strel('rectangle',[16 32]);
SE8 = strel('square',30);
SE9 = strel('cube',15);
SE10 = strel('cuboid',[15 15 15]);
SE11 = strel('sphere',15);

%% dilation
Id = imdilate(BW2,SE3);
figure('Name','dilation')
subplot(3,3,1), imshow(BW2,[]),title('Input Image');
subplot(3,3,2), imshow(imdilate(BW2,SE1),[]),title('SE1');
subplot(3,3,3), imshow(imdilate(BW2,SE2),[]),title('SE2');
subplot(3,3,4), imshow(imdilate(BW2,SE3),[]),title('SE3');
subplot(3,3,5), imshow(imdilate(BW2,SE4),[]),title('SE4');
subplot(3,3,6), imshow(imdilate(BW2,SE5),[]),title('SE5');
subplot(3,3,7), imshow(imdilate(BW2,SE6),[]),title('SE6');
subplot(3,3,8), imshow(imdilate(BW2,SE7),[]),title('SE7');
subplot(3,3,9), imshow(imdilate(BW2,SE8),[]),title('SE8');

%% erosion
Ie = imerode(BW2,SE3);
figure('Name','Erosion')
subplot(3,3,1), imshow(BW2,[]),title('Input Image');
subplot(3,3,2), imshow(imerode(BW2,SE1),[]),title('SE1');
subplot(3,3,3), imshow(imerode(BW2,SE2),[]),title('SE2');
subplot(3,3,4), imshow(imerode(BW2,SE3),[]),title('SE3');
subplot(3,3,5), imshow(imerode(BW2,SE4),[]),title('SE4');
subplot(3,3,6), imshow(imerode(BW2,SE5),[]),title('SE5');
subplot(3,3,7), imshow(imerode(BW2,SE6),[]),title('SE6');
subplot(3,3,8), imshow(imerode(BW2,SE7),[]),title('SE7');
subplot(3,3,9), imshow(imerode(BW2,SE8),[]),title('SE8');

%% Composing SEs
S11=[0 0 0; 1 1 1; 0 0 0]; S12=[0 1 0; 0 1 0; 0 1 0];
S1=imdilate(S11,S12);

S21=[0 0 0 0 0; 1 1 1 1 1; 0 0 0 0 0]; S22=[ 0 1 0 ; 0 1 0; 0 1 0];
S2=imdilate(S21,S22);

S31=[0 0 0 ; 1 0 0; 0 1 0]; S32=[ 0 0 1 ; 0 1 0; 0 0 0]; S33=[ 0 1 0 ; 0 1 0; 0 1 0];
S3=imdilate(imdilate(S31,S32,'full'), S33,'full');

S41=[0 0 0 ; 1 0 0; 0 1 0]; S42=[ 0 0 1 ; 0 1 0; 0 0 0]; 
S43=[ 0 0 0 ; 1 1 1; 0 0 0]; S44=[ 0 1 0 ; 0 1 0; 0 1 0]; 
S4=imdilate(imdilate(imdilate(S41,S42,'full'), S43,'full'), S44,'full');

S51=[1 0 0 ; 0 1 0; 0 0 1]; S52=[ 0 0 1 ; 0 1 0; 1 0 0]; 
S53=[ 0 0 0 ; 1 1 1; 0 0 0]; S54=[ 0 1 0 ; 0 1 0; 0 1 0]; 
S5=imdilate(imdilate(imdilate(S51,S52,'full'), S53,'full'), S54,'full');

%% Opeing
figure('Name','Opening (Structure Elements)')
subplot(3,3,1), imshow(BW2,[]),title('Input Image');
subplot(3,3,2), imshow(imopen(BW2,SE1),[]),title('SE1');
subplot(3,3,3), imshow(imopen(BW2,SE2),[]),title('SE2');
subplot(3,3,4), imshow(imopen(BW2,SE3),[]),title('SE3');
subplot(3,3,5), imshow(imopen(BW2,SE4),[]),title('SE4');
subplot(3,3,6), imshow(imopen(BW2,SE5),[]),title('SE5');
subplot(3,3,7), imshow(imopen(BW2,SE6),[]),title('SE6');
subplot(3,3,8), imshow(imopen(BW2,SE7),[]),title('SE7');
subplot(3,3,9), imshow(imopen(BW2,SE8),[]),title('SE8');

figure('Name','Opening (Composing SEs)')
subplot(3,2,1), imshow(BW2,[]),title('Input Image');
subplot(3,2,2), imshow(imopen(BW2,S1),[]),title('S1');
subplot(3,2,3), imshow(imopen(BW2,S2),[]),title('S2');
subplot(3,2,4), imshow(imopen(BW2,S3),[]),title('S3');
subplot(3,2,5), imshow(imopen(BW2,S4),[]),title('S4');
subplot(3,2,6), imshow(imopen(BW2,S5),[]),title('S5');

%% Closing
figure('Name','closing')
subplot(3,3,1), imshow(BW2,[]),title('Input Image');
subplot(3,3,2), imshow(imclose(BW2,SE1),[]),title('SE1');
subplot(3,3,3), imshow(imclose(BW2,SE2),[]),title('SE2');
subplot(3,3,4), imshow(imclose(BW2,SE3),[]),title('SE3');
subplot(3,3,5), imshow(imclose(BW2,SE4),[]),title('SE4');
subplot(3,3,6), imshow(imclose(BW2,SE5),[]),title('SE5');
subplot(3,3,7), imshow(imclose(BW2,SE6),[]),title('SE6');
subplot(3,3,8), imshow(imclose(BW2,SE7),[]),title('SE7');
subplot(3,3,9), imshow(imclose(BW2,SE8),[]),title('SE8');

%% Hit Miss operation
figure('Name','Hit Miss Operation')
subplot(2,2,1), imshow(BW2,[]),title('Input Image');
subplot(2,2,2), imshow(bwhitmiss(BW2,S11,S32),[]),title('S11 & S32');
subplot(2,2,3), imshow(bwhitmiss(BW2,S31,S32),[]),title('S31 % S32');
subplot(2,2,4), imshow(bwhitmiss(BW2,S51,S32),[]),title('S51 & S32');

%% Flood Fill
figure('Name','FloodFill')
subplot(1,3,1), imshow(BW2,[]),title('Input Image');
subplot(1,3,2), imshow(imfill(BW2,[150 150],4),[]),title('[150 150], 4-connected');
subplot(1,3,3), imshow(imfill(BW2,[200 200],8),[]),title('[200 200], 8-connected');


%% <Labelling/Classifying>
% Equivalence classes (Label connected components in 2-D binary image)
LE4 = bwlabel(BW2,4);
LE8 = bwlabel(BW2,8);

figure('Name','Labeling using equivalence classes')
subplot(1,3,1), imshow(BW2,[]), title('Input image');
subplot(1,3,2), imshow(bwlabel(BW2,4),[]),title('Labeling using 4-N');
subplot(1,3,3), imshow(bwlabel(BW2,8),[]),title('Labeling using 8-N');


% Union-find algorithm (Label connected components in binary image)
UF = bwlabeln(BW2);
UF4 = bwlabeln(BW2,4);
UF8 = bwlabeln(BW2,8);

figure('Name','Labeling using union-find algorithm')
subplot(1,3,1), imshow(BW2,[]), title('Input image');
subplot(1,3,2), imshow(UF4,[]),title('Labeling using 4-N');
subplot(1,3,3), imshow(UF8,[]),title('Labeling using 8-N');


% Connected components
CC4 = bwconncomp(BW2,4);
CC8 = bwconncomp(BW2,8); % default

CCL4 = labelmatrix(CC4);
CCL8 = labelmatrix(CC8);

figure('Name','Labeling using connected components algorithm')
subplot(1,3,1), imshow(BW2,[]),title('Input image');
subplot(1,3,2), imshow(CCL4,[]),title('Labeling using 4-N');
subplot(1,3,3), imshow(CCL8,[]),title('Labeling using 8-N');
