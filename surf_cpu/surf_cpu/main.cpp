#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace cv::detail;
using namespace cv::xfeatures2d;

float match_conf = 0.65f;

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
	GpuMat image_gpu_1, image_gpu_2, gray_image_gpu_1, gray_image_gpu_2;
	GpuMat keypoints_gpu_1, keypoints_gpu_2, descriptors_gpu_1, descriptors_gpu_2;
	Ptr<SURF> surf = SURF::create();

	vector<Mat> images;

	//600*800
	images.push_back(imread("image\\test1.jpg"));
	images.push_back(imread("image\\test2.jpg"));
	images.push_back(imread("image\\test1.jpg"));
	images.push_back(imread("image\\test2.jpg"));
	images.push_back(imread("image\\test1.jpg"));
	images.push_back(imread("image\\test2.jpg"));
	images.push_back(imread("image\\test1.jpg"));
	images.push_back(imread("image\\test2.jpg"));
	images.push_back(imread("image\\test1.jpg"));
	images.push_back(imread("image\\test2.jpg"));

	////1280*720
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));
	//images.push_back(imread("image\\image1.jpg"));
	//images.push_back(imread("image\\image2.jpg"));

	for (int n = 0; n < images.size(); n += 2)
	{
		int num = n / 2 + 1;
		UMat descriptors_1, descriptors_2, gray_image_1, gray_image_2;;
		vector<KeyPoint> keypoints_1, keypoints_2;
		MatchesInfo matches_info;
		set<pair<int, int>> matches;
		vector<vector<DMatch>> pair_matches;

		cout << "开始拼接第" << num << "张" << endl;
		int64 app_start_time = getTickCount();
		//寻找特征点
		cout << "特征点提取" << endl;
		int64 t = getTickCount();
		cv::cvtColor(images[n], gray_image_1, COLOR_BGR2GRAY);
		cv::cvtColor(images[n + 1], gray_image_2, COLOR_BGR2GRAY);
		UMat descriptors;
		surf->detectAndCompute(gray_image_1, Mat(), keypoints_1, descriptors);
		descriptors_1 = descriptors.reshape(1, (int)keypoints_1.size());
		surf->detectAndCompute(gray_image_2, Mat(), keypoints_2, descriptors);
		descriptors_2 = descriptors.reshape(1, (int)keypoints_2.size());

		cout << "特征点提取耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//特征点匹配
		cout << "特征点匹配" << endl;
		t = getTickCount();

		Ptr<cuda::DescriptorMatcher> matcher = cuda::DescriptorMatcher::createBFMatcher(NORM_L1);
		pair_matches.clear();
		descriptors_gpu_1.upload(descriptors_1);
		descriptors_gpu_2.upload(descriptors_2);
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
		//Mat match_image;
		//drawMatches(images[n], keypoints_1, images[n+1], keypoints_2, matches_info.matches, match_image);
		//string match_name = "match_image" + to_string(num) + ".jpg";
		//imwrite(match_name, match_image);



		//图像配准
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
		CalcCorners(matches_info.H, images[n + 1]);

		//图像配准  
		Mat imageTransform1, imageTransform2;
		cv::warpPerspective(images[n + 1], imageTransform2, matches_info.H, Size(MAX(corners.right_top.x, corners.right_bottom.x), images[n].rows));
		//imshow("直接经过透视矩阵变换", imageTransform2);
		//waitKey();
		//imwrite("trans_pic.jpg", imageTransform2);
		cout << "图像配准耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;


		//图像拼接
		cout << "图像拼接" << endl;
		t = getTickCount();
		//创建拼接后的图,需提前计算图的大小
		int dst_width = imageTransform2.cols;
		int dst_height = images[n].rows;

		Mat dst(dst_height, dst_width, CV_8UC3);
		dst.setTo(0);

		imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
		images[n].copyTo(dst(Rect(0, 0, images[n].cols, images[n].rows)));
		//imshow("dst_1", dst);
		//waitKey();
		//imwrite("dst1.jpg", dst);


		OptimizeSeam(images[n], imageTransform2, dst);
		//imshow("dst_2", dst);
		//waitKey();
		string file_name = "dst" + to_string(num) + ".jpg";
		imwrite(file_name, dst);
		cout << "图像拼接耗时: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
		cout << "总耗时: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec\n\n" << endl;
	}
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