clc;
clear all;
close all;

addpath("Images");
folder = 'Images\sequenceImages'; 
images = dir(fullfile(folder,'*.jpg'));
numImg = length(images);

for k = 1 : numImg
	fileName = fullfile(folder, images(k).name);
	img=imread(fileName);
	[row, column, color] = size(img);
	if k == 1  % 1 img
		sumImage = double(img);		
    else
		img = imresize(img, [row, column]);
		sumImage = sumImage + double(img);
		displayedImage = uint8(sumImage / k);
		%imshow(displayedImage, []);
	end
end

sumImage = uint8(sumImage / numImg);
%imshow(sumImage, []);
%imshow('Images\sequenceImages\sequenceImg (1).jpg');
CurrentFrame=imread('Images\sequenceImages\sequenceImg (1).jpg');

%Display Background and Foreground
subplot(2,2,1);imshow(sumImage);title('BackGround');
subplot(2,2,2);imshow(CurrentFrame);title('Current Frame');

%Convert RGB 2 HSV Color conversion
[Background_hsv]=round(rgb2hsv(sumImage));
[CurrentFrame_hsv]=round(rgb2hsv(CurrentFrame));
Out = bitxor(Background_hsv,CurrentFrame_hsv);

%Convert RGB 2 GRAY
Out=rgb2gray(Out);

%Read Rows and Columns of the Image
[rows columns]=size(Out);

%Convert to Binary Image
for i=1:rows
    for j=1:columns
        
        if Out(i,j) >0
            BinaryImage(i,j)=1;
        
        else
            BinaryImage(i,j)=0;
        end
    end
end


%Apply Median filter to remove Noise
FilteredImage=medfilt2(BinaryImage,[5 5]);


%Boundary Label the Filtered Image
[L num]=bwlabel(FilteredImage);

STATS=regionprops(L,'all');
cc=[];
removed=0;

%Remove the noisy regions
for i=1:num
    dd=STATS(i).Area;
    
    if (dd < 500)
        L(L==i)=0;
        removed = removed + 1;
        num=num-1;
    else
        
    end
end

[L2 num2]=bwlabel(L);

% Trace region boundaries in a binary image.

[B,L,N,A] = bwboundaries(L2);

%Display results

subplot(2,2,3),  imshow(L2);title('BackGround Detected');
subplot(2,2,4),  imshow(L2);title('Blob Detected');

hold on;

for k=1:length(B)
    if(~sum(A(k,:)))
        boundary = B{k};
        plot(boundary(:,2), boundary(:,1), 'r','LineWidth',2);
        
        for l=find(A(:,k))'
            boundary = B{l};
            plot(boundary(:,2), boundary(:,1), 'g','LineWidth',2);
        end
    end

end
