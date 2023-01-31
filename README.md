# computer_vision

## Environment

Computer Vision (2020 Fall Semester)
- Languages: MATLAB, Python


HW1 - Point Operations, Binary Image Processing
1. Set up a stationary camera and capture a short video sequence of some moving foreground objects while the background remains stationary. Compute the mean image of the sequence to generate a background image, then use background subtraction to segment the foreground objects. Display the foreground objects that result from thresholding the absolute difference between the current image(s) and the background image.
2. Identify several different types of small, readily available objects, and gather several instances of each. Examples might be coins, buttons, pencils, keys, and so forth. Place the objects on a single-colored table or floor, and take a picture that looks down on the scene. Write code to threshold the image, clean up the noise, label the components, compute various properties of the foreground regions, and automatically classify the regions according to the appropriate category. 


HW2 - Single Image Haze Removal Using Dark Channel Prior
 Apply the following filters on transmission maps t(x) obtained for different images
•	Bitonic Filter
•	Zero-Order Reverse Filtering
•	Guided Image Filtering
•	Weighted Least Squares based Filtering
•	Mutual-Structure for Joint Filtering
Compare the quantitative results (correlations and PNSR) for images given their ground truths
![image](https://user-images.githubusercontent.com/45842934/215753442-8af11453-32d3-4f27-8bf0-02fb69844b53.png)
1. Apply the following filters on transmission maps t(x) obtained for different images
• Bitonic Filter
• Zero-Order Reverse Filtering
• Guided Image Filtering
• Weighted Least Squares based Filtering
• Mutual-Structure for Joint Filtering
2. Compare the visual results for all images and include in report
3. Compare the qualitative results (correlation and PNSR) for images given their ground
truths and include in the report



Final Project
- Description: Image dehazing using different methods for different images
- Local Binary PAttern(LBP) + Sharpness Map + Hough Line Detection
- Preprocessing
Apply Gaussian filter  - noise removal
Kernel size: 3 x 3

Apply Laplacian filter
Depth: CV_16S
Kernel size: 3

Preprocessing is done to improve the quality for future blur map estimation

Motion blur images have lines stretching at one direction
Hough Line Detection
Apply inverse threshold to the image
-> Minimum line length + minimum number of lines + angle of lines


![image](https://user-images.githubusercontent.com/45842934/215755083-72d247f7-ee7c-4583-8705-49677b6dc275.png)
Blur Map
![image](https://user-images.githubusercontent.com/45842934/215755480-57d018d4-619d-47b2-97a1-8a4197bc1ab3.png)
