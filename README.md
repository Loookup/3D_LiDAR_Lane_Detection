# 3D_LiDAR_Lane_Detection

3D LiDAR based 3 Degree poly-fitted Lane Detection


## Introduction


In autonomous driving, as known as self-driving, it requires a lot of technologies, especially perceiving near environment and extracting proper information is most important part. Camera and LiDAR are commonly used for perception and each has different advantages and disadvantages. Camera has a lot of information, so we can extract colors, detect objects, track object, etc.. But with monocular, there’s no way to get the distance between desired object and camera. But LiDAR gives exact distance and shape of object. Also, LiDAR gives robust data compare with camera in bad weather. So, if you want to detect lane in road, LiDAR might be better option, because you can easily detect lane. Therefore, that’s why intensity is critical value in lane detection when using LiDAR.


## Methodology


### 1. Preprocess point cloud data
After getting point cloud data from LiDAR data, multiply by 10 to intensity for scale up. And find road plane with RANSAC(RANdom SAmple Consensus) method. If points belong to same plane, their groups must be have similar normal planes. And also, since RANSAC eliminates outlier of group, so, it ensures robustness. After 100 times of iterations, extract road plane and threshold intensity with specific number to divide road points and lane points.


### 2. Get starting point of lane with histogram
After preprocessing, although the lanes are shown quite intuitively, it’s hard to decide which one is the right next lane of vehicle. So, use histogram of intensity near the origin coordinate aligned with Y-axis. But, if you designates whole range of Y, the result will be unreliable if lanes are curved. So, set ROI(region of Interest) near origin and avoid redundant errors.
After getting histogram, decide which lanes are left and right lane next to origin coordinate and get starting points in cartesian coordinate for each lane.


### 3. Detect Lane using Sliding Window
After getting starting points on Y-axis, detect lanes with sliding window method. Sliding window method makes boxes(bins) aligned with designated axis and moves to highest value which is placed inside each boxes. So, set ROI with range 10 ~ 60 in X axis and divided with 30 boxes. But, there’re many boxes which have more than one points with highest value, so calculate mean points and update each boxes. It gives great result and get final points on lanes.


### 4. Fit Lane with 3rd degree polynomial
After getting lane points, smoothing lane with high order of polynomial because self-driving needs path planning on autonomous navigation. But, if it’s too high, it requires complexity and makes real-time navigation hard. So use 3rd degree polynomial and fit with Numpy package and get 4 coefficients of polynomial.


### 5. Save Coefficients and Load Visualize.py
After getting coefficients of 3rd degree polynomial, save them with txt format with ‘savetxt’ function of numpy package like sample output format.
