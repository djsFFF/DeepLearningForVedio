#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv_modules.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cv::detail;

vector<String> img_names;
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
string estimator_type = "homography";
string ba_cost_func = "ray";
string ba_refine_mask = "xxxxx";
bool do_wave_correct = true;
WaveCorrectKind wave_correct = detail::WAVE_CORRECT_HORIZ;
bool save_graph = false;
std::string save_graph_to;
string warp_type = "spherical";
int expos_comp_type = ExposureCompensator::GAIN_BLOCKS;
float match_conf = 0.65f;
string seam_find_type = "gc_color";
int blend_type = Blender::MULTI_BAND;
int timelapse_type = Timelapser::AS_IS;
float blend_strength = 5;
string result_name = "result.jpg";
bool timelapse = false;
int range_width = -1;

static int parseArgs()
{
	if (preview)
	{
		compose_megapix = 0.6;
	}
}

void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst);

typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}four_corners_t;

four_corners_t corners;

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

int main(void)
{
	int64 app_start_time = getTickCount();
	img_names.push_back("image\\test1.jpg");
	img_names.push_back("image\\test2.jpg");
	int num_images = static_cast<int>(img_names.size());
	if (num_images < 2)
	{
		cout << "Need more images" << endl;
		return -1;
	}
	double work_scale = 1;
	bool is_work_scale_set = false;

	//寻找特征点
	cout << "特征点提取" << endl;
	int64 t = getTickCount();
	Ptr<FeaturesFinder> finder;
	if (try_cuda)
		finder = makePtr<SurfFeaturesFinderGpu>();
	else
		finder = makePtr<SurfFeaturesFinder>();


	Mat full_img;
	vector<ImageFeatures> features(num_images);
	vector<Mat> images(num_images);
	vector<Size> full_img_sizes(num_images);
	double seam_work_aspect = 1;
	for (int i = 0; i < num_images; i++)
	{
		//full_img = imread(img_names[i]);
		////imshow("原图", full_img);
		//full_img_sizes[i] = full_img.size();
		//if (full_img.empty())
		//{
		//	cout << "Can't open image " + img_names[i] << endl;
		//	return -1;
		//}

		//if (work_megapix < 0)
		//{
		//	img = full_img;
		//	work_scale = 1;
		//	is_work_scale_set = true;
		//}
		//else
		//{
		//	if (!is_work_scale_set)
		//	{
		//		work_scale = min(1.0, sqrt(work_megapix * 1e6 / full_img.size().area()));
		//		is_work_scale_set = true;
		//	}
		//	//面积大于work_megapix*100000的图片，长宽调整为原来的work_scale倍
		//	resize(full_img, img, Size(), work_scale, work_scale, INTER_LINEAR_EXACT);
		//	//imshow("压缩1", img);
		//}

		////计算特征点
		//(*finder)(img, features[i]);
		//features[i].img_idx = i;
		//cout << "Features in image #" << i + 1 << ": " << features[i].keypoints.size() << endl;
		//images[i] = img.clone();

		full_img = imread(img_names[i]);
		(*finder)(full_img, features[i]);
		features[i].img_idx = i;
		images[i] = full_img.clone();
	}
	finder->collectGarbage();
	full_img.release();
	//img.release();
	cout << "特征点提取耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

	//图像匹配
	cout << "特征点匹配" << endl;
	t = getTickCount();
	vector<MatchesInfo> pairwise_matches;
	Ptr<FeaturesMatcher> matcher;
	if (matcher_type == "affine")
	{
		matcher = makePtr<AffineBestOf2NearestMatcher>(false, try_cuda, match_conf);
	}
	else if (range_width == -1)
	{
		//最近邻和次近邻法
		matcher = makePtr<BestOf2NearestMatcher>(try_cuda, match_conf);
	}
	else
	{
		matcher = makePtr<BestOf2NearestRangeMatcher>(range_width, try_cuda, match_conf);
	}
	(*matcher)(features, pairwise_matches);
	matcher->collectGarbage();
	cout << "特征点匹配耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

	Mat first_match;
	drawMatches(images[0], features[0].keypoints, images[1], features[1].keypoints, pairwise_matches[1].matches, first_match);
	//imshow("first_match ", first_match);
	//waitKey();

	cout << "图像配准" << endl;
	t = getTickCount();
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i < pairwise_matches[1].matches.size(); i++)
	{
		imagePoints1.push_back(features[0].keypoints[pairwise_matches[1].matches[i].queryIdx].pt);
		imagePoints2.push_back(features[1].keypoints[pairwise_matches[1].matches[i].trainIdx].pt);
	}

	//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
	Mat homo = findHomography(imagePoints2, imagePoints1, CV_RANSAC);
	//输出映射矩阵   
	//cout << "变换矩阵为：\n" << homo << endl << endl;    

	//计算配准图的四个顶点坐标
	CalcCorners(homo, images[1]);
	//cout << "left_top:" << corners.left_top;
	//cout << " left_bottom:" << corners.left_bottom;
	//cout << " right_top:" << corners.right_top;
	//cout << " right_bottom:" << corners.right_bottom;

	//图像配准  
	Mat imageTransform1, imageTransform2;
	warpPerspective(images[1], imageTransform2, homo, Size(MAX(corners.right_top.x, corners.right_bottom.x), images[0].rows));
	cout << "图像配准耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
	//imshow("直接经过透视矩阵变换", imageTransform2);
	//waitKey();
	imwrite("trans_pic.jpg", imageTransform2);

	cout << "图像拼接" << endl;
	t = getTickCount();
	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform2.cols;  //取最右点的长度为拼接图的长度
	int dst_height = images[0].rows;

	Mat dst(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);

	imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
	images[0].copyTo(dst(Rect(0, 0, images[0].cols, images[0].rows)));

	//imshow("b_dst", dst);
	//waitKey();
	imwrite("dst1.jpg", dst);


	OptimizeSeam(images[0], imageTransform2, dst);
	cout << "图像拼接耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

	//imshow("dst", dst);
	imwrite("dst2.jpg", dst);
	cout << "总耗时: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec" << endl;
	getchar();

	return 0;
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