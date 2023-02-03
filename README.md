# Computer Vision

## Environment
- Computer Vision (2020 Fall Semester)
- Languages: MATLAB, Python


## Description
### CV01
#### Point Operations
- Short video sequence of some moving foreground objects while the background remains stationary
- Generate a backround image with the mean image of the sequence
- Use background subtraction to segment the foreground objects <br/>
<img src="https://user-images.githubusercontent.com/45842934/216033818-e3b7a9b4-932b-40a3-98f7-514dac254b84.png" height="260" />

#### Binary Image Processing 
- Threshold the image, clean up the noise, compute various properties of the foreground regions
<img src="https://user-images.githubusercontent.com/45842934/216032753-4ea3a746-1c70-451e-a5b7-1304bc91f85b.png" height="320" />

---

### CV02
#### Single Image Haze Removal Using Dark Channel Prior
 Apply various filters on transmission maps t(x) obtained for different images and compare visual and qualitative results (correlation and PNSR)
- Bitonic Filter
- Zero-Order Reverse Filtering
- Guided Image Filtering
- Weighted Least Squares based Filtering
- Mutual-Structure for Joint Filtering

![image](https://user-images.githubusercontent.com/45842934/215753442-8af11453-32d3-4f27-8bf0-02fb69844b53.png)


---

### CV03
#### Image dehazing using different methods for different images
- Local Binary PAttern(LBP) + Sharpness Map + Hough Line Detection
- Preprocessing: to improve the quality for future blur map estimation
- Apply Gaussian filter  - noise removal
- Kernel size: 3 x 3
- Depth: CV_16S

For detecting motion blurs, images have lines stretching at one direction. Apply inverse threshold to the image. <br/>
-> Minimum line length + minimum number of lines + angle of lines

Motion Blur <br/>
<img src="https://user-images.githubusercontent.com/45842934/215755083-72d247f7-ee7c-4583-8705-49677b6dc275.png" height="200" />
<br/>
Blur Map <br/>
<img src="https://user-images.githubusercontent.com/45842934/215755480-57d018d4-619d-47b2-97a1-8a4197bc1ab3.png" height="310" />
