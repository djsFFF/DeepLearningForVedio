#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace cv::detail;

bool preview = false;
bool try_cuda = true;
//图像，面积为work_megapix*100000
double work_megapix = 0.6;
//拼接缝像素的大小
double seam_megapix = 0.1;
double compose_megapix = -1;
float conf_thresh = 1.f;
string features_type = "surf";
string matcher_type = "homography";
float match_conf = 0.65f;
int range_width = -1;

static int parseArgs()
{
	if (preview)
	{
		compose_megapix = 0.6;
	}
}

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);
void CalcCorners(const Mat& H, const Mat& src);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;
four_corners_t corners;

int main(void)
{
	int64 app_start_time = getTickCount();
	Mat image_1 = imread("E:\\opencv\\my_stitching\\my_stitching\\image\\test1.jpg");
	Mat image_2 = imread("E:\\opencv\\my_stitching\\my_stitching\\image\\test2.jpg");

	//寻找特征点
	cout << "特征点提取" << endl;
	int64 t = getTickCount();
	
	GpuMat image_gpu_1, image_gpu_2, gray_image_gpu_1, gray_image_gpu_2;
	GpuMat keypoints_gpu_1, keypoints_gpu_2, descriptors_gpu_1, descriptors_gpu_2;
	UMat descriptors_1, descriptors_2;
	vector<KeyPoint> keypoints_1, keypoints_2;
	SURF_CUDA surf;

	CV_Assert(image_1.depth() == CV_8U);
	CV_Assert(image_2.depth() == CV_8U);
	image_gpu_1.upload(image_1);
	image_gpu_2.upload(image_2);
	cv::cuda::cvtColor(image_gpu_1, gray_image_gpu_1, COLOR_BGR2GRAY);
	cv::cuda::cvtColor(image_gpu_2, gray_image_gpu_2, COLOR_BGR2GRAY);
	surf.nOctaves = 3;
	surf.nOctaveLayers = 4;
	surf.upright = false;
	surf(gray_image_gpu_1, GpuMat(), keypoints_gpu_1);
	surf(gray_image_gpu_2, GpuMat(), keypoints_gpu_2);

	surf.nOctaves = 4;
	surf.nOctaveLayers = 2;
	surf.upright = true;
	surf(gray_image_gpu_1, GpuMat(), keypoints_gpu_1, descriptors_gpu_1, true);
	surf(gray_image_gpu_2, GpuMat(), keypoints_gpu_2, descriptors_gpu_2, true);

	surf.downloadKeypoints(keypoints_gpu_1, keypoints_1);
	surf.downloadKeypoints(keypoints_gpu_2, keypoints_2);
	descriptors_gpu_1.download(descriptors_1);
	descriptors_gpu_2.download(descriptors_2);

	image_gpu_1.release();
	image_gpu_2.release();
	gray_image_gpu_1.release();
	gray_image_gpu_2.release();
	keypoints_gpu_1.release();
	keypoints_gpu_2.release();
	descriptors_gpu_1.release();
	descriptors_gpu_2.release();
	surf.releaseMemory();

	cout << "特征点提取耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

	//图像匹配
	cout << "特征点匹配" << endl;
	t = getTickCount();
	MatchesInfo matches_info;
	vector<vector<DMatch>> pair_matches;
	set<pair<int, int>> matches;
	descriptors_gpu_1.upload(descriptors_1);
	descriptors_gpu_2.upload(descriptors_2);
	Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);

	pair_matches.clear();
	matcher->knnMatch(descriptors_gpu_1, descriptors_gpu_2, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); i++)
	{
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf) * m1.distance)
		{
			matches_info.matches.push_back(m0);
			matches.insert(make_pair(m0.queryIdx, m0.trainIdx));
		}
	}

	pair_matches.clear();
	matcher->knnMatch(descriptors_gpu_2, descriptors_gpu_1, pair_matches, 2);
	for (size_t i = 0; i < pair_matches.size(); ++i)
	{
		if (pair_matches[i].size() < 2)
			continue;
		const DMatch& m0 = pair_matches[i][0];
		const DMatch& m1 = pair_matches[i][1];
		if (m0.distance < (1.f - match_conf) * m1.distance)
			if (matches.find(std::make_pair(m0.trainIdx, m0.queryIdx)) == matches.end())
				matches_info.matches.push_back(DMatch(m0.trainIdx, m0.queryIdx, m0.distance));
	}

	descriptors_gpu_1.release();
	descriptors_gpu_2.release();
	vector<vector<DMatch>>().swap(pair_matches);
	matcher.release();
	cout << "特征点匹配耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
	//Mat first_match;
	//drawMatches(image_1, keypoints_1, image_2, keypoints_2, matches_info.matches, first_match);
	//imshow("first_match ", first_match);
	//waitKey();

	cout << "图像配准" << endl;
	t = getTickCount();
	Mat src_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	Mat dst_points(1, static_cast<int>(matches_info.matches.size()), CV_32FC2);
	for (size_t i = 0; i < matches_info.matches.size(); i++)
	{
		const DMatch& m = matches_info.matches[i];

		Point2f p = keypoints_2[m.trainIdx].pt;
		src_points.at<Point2f>(0, static_cast<int>(i)) = p;

		p = keypoints_1[m.queryIdx].pt;
		dst_points.at<Point2f>(0, static_cast<int>(i)) = p;
	}
	matches_info.H = findHomography(src_points, dst_points, RANSAC);

	//计算配准图的四个顶点坐标
	CalcCorners(matches_info.H, image_2);

	//图像配准  
	Mat imageTransform1, imageTransform2;
	cv::warpPerspective(image_2, imageTransform2, matches_info.H, Size(MAX(corners.right_top.x, corners.right_bottom.x), image_1.rows));
	cout << "图像配准耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
	//imshow("直接经过透视矩阵变换", imageTransform2);
	//waitKey();
	imwrite("trans_pic.jpg", imageTransform2);

	cout << "图像拼接" << endl;
	t = getTickCount();
	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform2.cols;  
	int dst_height = image_1.rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
	image_1.copyTo(dst(Rect(0, 0, image_1.cols, image_1.rows)));
	//imshow("dst_1", dst);
	//waitKey();
	imwrite("dst1.jpg", dst);


	OptimizeSeam(image_1, imageTransform2, dst);
	cout << "图像拼接耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
	//imshow("dst_2", dst);
	//waitKey();
	imwrite("dst2.jpg", dst);
	cout << "总耗时: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	getchar();

	return 0;
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = H * V2;
	//左上角(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

//优化两图的连接处，使得拼接自然
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = img1.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = img1.cols; //注意，是列数*通道数
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img1中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成正比，实验证明，这种方法确实好  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}