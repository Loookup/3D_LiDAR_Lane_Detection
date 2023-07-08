'''
This Script was written with following software specification, packages

    Python 3.8.10
    Ubuntu 20.04 LTS
    PCL 1.10.0
    Numpy 1.24.3
    Matplotlib 3.1.2
    
Command terminal : $ python3 test.py
Load point cloud datas from pointclouds folder
lane coefficients will be saved in sample_output folder
'''
import numpy as np
import pcl
import pcl.pcl_visualization
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
import copy
import math
from scipy.signal import find_peaks
import numpy.polynomial.polynomial as poly


class Lane:

    def __init__(self, _file_num):

        self.file_num = _file_num   # LiDAR Data file Number

        self.bin_file = self.file_num + ".bin"  # LiDAR Data bin file

        self.points = np.fromfile("./pointclouds/"+self.bin_file, dtype=np.float32).reshape(-1, 5)   # (x,y,z,intensity)

        self.points_xyzi = copy.deepcopy(self.points[:, :4])    # Keep origin

        self.scale_factor = 10      # Scale up Intensity, Default = 10

        self.num_lanes = 2          # Left, Right Lane >> 2 Lanes

        self.bin_hist_num = 30      # Number of bins of histogram, Default = 30

        self.ROI_hist_z_thresh = 0.5   # Z-axis threshold ROI(Region of Interest) for histogram, Default = 0.5

        self.ROI_hist_min = 7.0     # ROI(Region of Interest) for histogram, Default = 7.0

        self.ROI_hist_max = 20.0    # ROI(Region of Interest) for histogram, Default = 20.0

        self.ROI_Det_min = 10.0     # ROI(Region of Interest) for Sliding Window, Default = 10.0

        self.ROI_Det_max = 60.0     # ROI(Region of Interest) for Sliding Window, Default = 60.0

        self.bin = 30               # Number for Sliding Window Bin, Default = 30.0

        self.order = 3              # Order of Polynomial fitting

        self.offset = 0.35          # y size of Sliding Window Bin, Default = 0.35

        self.data_folder = "pointclouds"        # LiDAR Data bin file Folder

        self.lane_folder = "sample_output"        # Lane Coefficient of Fitted Lane Folder


    def Pipeline(self):
        '''
        Choose True or False for each Visualization Operation
        '''
        self.Max_and_Min()

        self.mulitple_scale(self.scale_factor)

        road_plane, road_plane_idx = self.find_road_plane(self.points_xyzi)   # Default = ON

        line = self.intensity_thresholding(road_plane, threshold=30)      # Default = 30

        self.Visualization(line)

        show_peaks = True   # Visualization True or False

        front_lanes, back_lanes = self.get_front_back_lane(line, plot_visual=show_peaks)

        show_detected_left_right_lane = True    # Visualization True or False

        sorted_left_lanes, sorted_right_lanes = self.frontback_to_leftright_lane(front_lanes, back_lanes, plot_visual=show_detected_left_right_lane)

        show_fitted_lane = True     # Visualization True or False

        left_coeff, right_coeff = self.get_fitted_lane(sorted_left_lanes, sorted_right_lanes, plot_visual=show_fitted_lane)

        is_save = True      # Save True or False

        self.save_lane_coeff_txt(left_coeff, right_coeff, is_save=is_save)


    def Max_and_Min(self):
        '''
        print range of point cloud data
        '''
        print ("Min X:{:15.2f}, Max X:{:15.2f}".format(np.min(self.points[:,0]), np.max(self.points[:,0])))
        print ("Min Y:{:15.2f}, Max Y:{:15.2f}".format(np.min(self.points[:,1]), np.max(self.points[:,1])))
        print ("Min Z:{:15.2f}, Max Z:{:15.2f}".format(np.min(self.points[:,2]), np.max(self.points[:,2])))
        print ("Min I:{:15.2f}, Max I:{:15.2f}".format(np.min(self.points[:,3]), np.max(self.points[:,3])))


    def mulitple_scale(self, scale_factor):
        '''
        for scale of Intensity of point cloud data
        '''
        scaled_intensity = np.clip(self.points_xyzi[:, 3] * scale_factor, 0, 255).astype(np.float32)

        self.points_xyzi[:, 3] = scaled_intensity


    def find_road_plane(self, xyz_data):
        '''
        Find Road(Normal) Plane with RANSAC(RANdom SAmple Consensus)
        '''
        cloud = pcl.PointCloud_PointXYZI()

        cloud.from_array(xyz_data.astype('float32'))

        seg = cloud.make_segmenter_normals(ksearch=50)

        seg.set_optimize_coefficients(True)

        seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)

        seg.set_normal_distance_weight(0.001)

        seg.set_method_type(pcl.SAC_RANSAC)

        seg.set_max_iterations(100)

        seg.set_distance_threshold(0.3)

        indices, model = seg.segment()

        cloud_plane = cloud.extract(indices, negative=False)

        return cloud_plane.to_array(), np.array(indices)


    def SetROI(self, orig_points, xmin, xmax, ymin, ymax, zmin, zmax, imin, imax):
        '''
        Set ROI(Region of Interest) for Real-time Autonomous Navigation
        '''
        points = copy.deepcopy(orig_points)

        if xmin != None and xmax != None:

            points = points[np.logical_and(points[:,0] > xmin, points[:,0] < xmax)]

        if ymin != None and ymax != None:

            points = points[np.logical_and(points[:,1] > ymin, points[:,1] < ymax)]

        if zmin != None and zmax != None:

            points = points[np.logical_and(points[:,2] > zmin, points[:,2] < zmax)]

        if imin != None and imax != None:

            points = points[np.logical_and(points[:,3] > imin, points[:,3] < imax)]

        return points


    def Visualization(self, points):
        '''
        Visualize Point Cloud Data with PCL(Point Cloud Library)
        '''
        cloud = pcl.PointCloud_PointXYZI()

        cloud.from_array(points.astype('float32'))

        visual = pcl.pcl_visualization.CloudViewing()

        visual.ShowGrayCloud(cloud, b'cloud')

        v = True

        while v:

            v = not(visual.WasStopped())


    def intensity_thresholding(self, points, threshold):
        '''
        Thresholding Intensity
        '''
        return points[np.logical_and(points[:,3]>threshold, points[:,3]<256)]


    def peak_intensity_ratio(self, orig_points, indicator, plot_visual):
        '''
        Find the peaks of Intensity based Y - Axis to Detect start of lane from current position
        So, Set ROI nearby current position
        '''
        if indicator == "front":
        
            points = self.SetROI(orig_points, self.ROI_hist_min, self.ROI_hist_max, None, None, -self.ROI_hist_z_thresh, self.ROI_hist_z_thresh, None, None) # Default = None
        
        elif indicator == "back":

            points = self.SetROI(orig_points, -self.ROI_hist_max, -self.ROI_hist_min, None, None, -self.ROI_hist_z_thresh, self.ROI_hist_z_thresh, None, None) # Default = None

        else:
            print("Unexpected Input, corrupted")
            return 0

        self.Visualization(points)        # Visualize ROI

        y = points[:,1]

        min_y = math.ceil(y.min())
        max_y = math.ceil(y.max())

        y_val = np.linspace(min_y, max_y, self.bin_hist_num)

        avg_intensity=[]
        ymean=[]

        for i in range(len(y_val)-1):

            indices = [k for k in range(len(y)) if y[k]>=y_val[i] and y[k]<y_val[i+1]]
 
            intensity_sum=0

            for j in indices:

                intensity_sum+= points[j,3]

            avg_intensity.append(intensity_sum)

            ymean.append((y_val[i]+y_val[i+1])/2)

        peaks, _ = find_peaks(avg_intensity, height=0)

        peaks = peaks.astype(np.int8)

        left, right = self.find_nearest_peak(ymean, peaks)

        if plot_visual:

            plt.title(indicator)
                
            plt.plot(ymean, avg_intensity,'--k')

            for peak in peaks:

                plt.plot(ymean[peak], avg_intensity[peak], "x")

            plt.plot(ymean[peaks[left]], avg_intensity[peaks[left]], "ro")

            plt.plot(ymean[peaks[right]], avg_intensity[peaks[right]], "ro")

            plt.show()

        else: pass

        peaks = [ymean[peaks[left]], ymean[peaks[right]]]

        return peaks


    def find_nearest_peak(self, ymean, peaks):
        '''
        Find Nearest 2 Peaks of (0), that points must be start points of Left and Right Lane 
        '''
        right_peak = []
        left_peak = []
        listlist = []

        for peak in peaks:

            listlist.append(ymean[peak])

            if ymean[peak] > 0:

                right_peak.append(ymean[peak])
            
            else:

                left_peak.append(ymean[peak])

        right = np.where(listlist == np.min(right_peak))

        left = np.where(listlist == np.max(left_peak))

        return left[0][0], right[0][0]


    def DetectLanes_Sliding_Window(self, data, startLanePoints, indicator, plot_visual):
        '''
        Use Sliding Window starts from nearest peaks of intensity histogram 
        '''
        lanes = np.zeros((self.bin, 2, self.num_lanes))

        if indicator == "front":

            laneStartX = np.linspace(self.ROI_Det_min, self.ROI_Det_max, self.bin)

        elif indicator == "back":

            laneStartX = np.linspace(-self.ROI_Det_min, -self.ROI_Det_max, self.bin)

        else:
            print("Unexpected Input, corrupted")
            return 0
                
        for j in range(self.num_lanes):

            laneStartY = startLanePoints[j]

            offset = self.offset

            for i in range(self.bin):
                    
                if indicator == "front":

                    indices = np.where((data[:,0] < laneStartX[i] +  (self.ROI_Det_max - self.ROI_Det_min)/self.bin) & (data[:,0] >= laneStartX[i]) &
                                    (data[:,1] < laneStartY + offset) & (data[:,1] >= laneStartY - offset))[0]
                    
                elif indicator == "back":

                    indices = np.where((data[:,0] < laneStartX[i]) & (data[:,0] >= laneStartX[i] - (self.ROI_Det_max - self.ROI_Det_min)/self.bin) &
                                    (data[:,1] < laneStartY + offset) & (data[:,1] >= laneStartY - offset))[0]

                if len(indices)!=0:
                
                    roi_data=data[indices,:]

                    max_intensity=np.argmax(roi_data[:,3])

                    idx = np.where(roi_data[:,3] == roi_data[max_intensity,3])

                    # If there're multiple points which have maximun intensity, then get mean point of them
                    x_mean = np.sum(roi_data[idx, 0]) / len(idx[0])

                    y_mean = np.sum(roi_data[idx, 1]) / len(idx[0])

                    val= [x_mean, y_mean]
            
                    lanes[i,:,j] = val

                    laneStartY = y_mean

                    if plot_visual:

                        plt.scatter(lanes[i,:,j][0], lanes[i,:,j][1], c='red')

                    else: pass

                else: pass
                    
        return lanes
    

    def get_front_back_lane(self, line, plot_visual):
        '''
        Get Front Lane and Back Lane based on points from Sliding Window Method
        '''
        front_peaks = self.peak_intensity_ratio(line, "front", plot_visual)

        back_peaks = self.peak_intensity_ratio(line, "back", plot_visual)

        front_lanes = self.DetectLanes_Sliding_Window(line, front_peaks, "front", plot_visual)

        back_lanes = self.DetectLanes_Sliding_Window(line, back_peaks, "back", plot_visual)

        if plot_visual:

            plt.title("Non-fitted lane")
            plt.axes().set_aspect('equal')
            plt.xlim(-self.ROI_Det_max, self.ROI_Det_max)
            plt.ylim(-10,10)
            plt.show()

        else: pass


        return front_lanes, back_lanes
    

    def frontback_to_leftright_lane(self, front_lanes, back_lanes, plot_visual):
        '''
        Get Left and Right Lane from Front and Back Lane
        '''
        left_lanes = np.vstack([front_lanes[:,:,1], back_lanes[:,:,1]])

        left_lanes = left_lanes[np.logical_and(left_lanes[:,1] != 0, left_lanes[:,1] != 0)]

        right_lanes = np.vstack([front_lanes[:,:,0], back_lanes[:,:,0]])

        right_lanes = right_lanes[np.logical_and(right_lanes[:,1] != 0, right_lanes[:,1] != 0)]

        sort_col_left = left_lanes[:, 0]

        idx_left = np.argsort(sort_col_left)

        sorted_left_lanes = left_lanes[idx_left]

        sort_col_right = right_lanes[:, 0]

        idx_right = np.argsort(sort_col_right)

        sorted_right_lanes = right_lanes[idx_right]

        if plot_visual:

            plt.plot(sorted_left_lanes[:,0], sorted_left_lanes[:,1], "r")
            plt.plot(sorted_right_lanes[:,0], sorted_right_lanes[:,1], "b")
            plt.title("Non-fitted lane")
            plt.axes().set_aspect('equal')
            plt.xlim(-self.ROI_Det_max,self.ROI_Det_max)
            plt.ylim(-10,10)
            plt.show()

        else: pass

        return sorted_left_lanes, sorted_right_lanes
    

    def get_fitted_lane(self, sorted_left_lanes, sorted_right_lanes, plot_visual):
        '''
        Get Fitted Lane using Numpy Polynomial (3rd)
        '''
        left_fitted_lane, left_coeff= self.fit_polynomial(sorted_left_lanes)

        right_fitted_lane, right_coeff = self.fit_polynomial(sorted_right_lanes)

        if plot_visual:

            plt.title("fitted lane")

            for dot in left_fitted_lane:
            
                plt.plot(dot[:,0],dot[:,1],color='r')

            for dot in right_fitted_lane:
            
                plt.plot(dot[:,0],dot[:,1],color='b')

            plt.axes().set_aspect('equal')
            plt.xlim(-self.ROI_Det_max, self.ROI_Det_max)
            plt.ylim(-10,10)
            plt.show()

        else: pass

        return left_coeff, right_coeff


    def fit_polynomial(self, lane):
        ''' 
        estimates a polynomial curve on the detected lane points 
        '''
        fitted_lane =[]
        x = lane[:,0]
        y = lane[:,1]
        
        if len(x)>0 and len(y)>0:

            coefs = poly.polyfit(x, y, self.order)
             
            fitted_x = x

            ffit = poly.Polynomial(coefs)

            fitted_y = ffit(fitted_x)

            point=np.concatenate((fitted_x.reshape(-1,1),fitted_y.reshape(-1,1)),axis=1)

            fitted_lane.append(point)

        return fitted_lane, coefs
    

    def save_lane_coeff_txt(self, left_coeff, right_coeff, is_save):
        '''
        Save coefficients of 3rd degree ploynomial with txt format
        '''
        txt = self.file_num + ".txt"

        lane_coeffi = np.array([[left_coeff[3], left_coeff[2], left_coeff[1], left_coeff[0]], [right_coeff[3], right_coeff[2], right_coeff[1], right_coeff[0]]])

        print("lane coeffi : " + str(lane_coeffi))

        if is_save:

            np.savetxt("./sample_output/"+txt, lane_coeffi, fmt='%.18e', delimiter=';')

        else: pass
    

if __name__ == "__main__":

    file_1 = "1553565729015329642"      # Default >> Succeed

    file_2 = "1553565776121143870"      # threshold = 0, No Road Plane >> Succeed

    file_3 = "1553565884136495856"      

    file_4 = "1553567105504169477"      

    file_5 = "1553669108359991937"      

    file_6 = "1553670562447716965"      

    file_7 = "1553670669147427892"      

    file_8 = "1553670684146333606"      

    # Choose desired data file

    my_Lane = Lane(file_1)

    my_Lane.Pipeline()
