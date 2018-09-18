#include <iostream>
#include <fstream>
#include <string>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;
using namespace cuda;
using namespace cv::detail;

float match_conf = 0.65f;
//ORB����
Size grid_size = Size(3, 1);
int nfeatures = 1500;
float scaleFactor = 1.3f;
//Ĭ��Ϊ5���޸�Ϊ1
int nlevels = 1;

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
	GpuMat descriptors_gpu_1, descriptors_gpu_2;

	vector<Mat> images;

	//600*800
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test1.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test2.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test1.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test2.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test1.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test2.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test1.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test2.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test1.jpg"));
	images.push_back(imread("E:\\opencv_git\\orb_cpu\\orb_cpu\\image\\test2.jpg"));


	////1280*720
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image1.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image2.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image1.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image2.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image1.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image2.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image1.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image2.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image1.jpg"));
	//images.push_back(imread("E:\\opencv_git\\orb_gpu\\orb_cpu\\image\\image2.jpg"));


	for (int n = 0; n < images.size(); n += 2)
	{
		int num = n / 2 + 1;
		UMat descriptors_1, descriptors_2;
		vector<KeyPoint> keypoints_1, keypoints_2;
		MatchesInfo matches_info;
		set<pair<int, int>> matches;
		vector<vector<DMatch>> pair_matches;

		cout << "��ʼƴ�ӵ�" << num << "��" << endl;
		int64 app_start_time = getTickCount();

		//Ѱ��������
		cout << "��������ȡ" << endl;
		int64 t = getTickCount();
		Ptr<cv::ORB> orb = cv::ORB::create(nfeatures * (99 + grid_size.area()) / 100 / grid_size.area(), scaleFactor, nlevels);
		UMat gray_image_1, gray_image_2;
		cv::cvtColor(images[n], gray_image_1, COLOR_BGR2GRAY);
		cv::cvtColor(images[n + 1], gray_image_2, COLOR_BGR2GRAY);
		////GPU�Ҷ�ͼת��
		//image_gpu_1.upload(images[n]);
		//image_gpu_2.upload(images[n + 1]);
		//cv::cuda::cvtColor(image_gpu_1, gray_image_gpu_1, COLOR_BGR2GRAY);
		//cv::cuda::cvtColor(image_gpu_2, gray_image_gpu_2, COLOR_BGR2GRAY);
		//gray_image_gpu_1.download(gray_image_1);
		//gray_image_gpu_2.download(gray_image_1);
		std::vector<KeyPoint> points;
		Mat _descriptors;
		UMat descriptors;

		//��ȡ��һ��ͼƬ�������㼰����
		for (int r = 0; r < grid_size.height; ++r)
			for (int c = 0; c < grid_size.width; ++c)
			{
				int xl = c * gray_image_1.cols / grid_size.width;
				int yl = r * gray_image_1.rows / grid_size.height;
				int xr = (c + 1) * gray_image_1.cols / grid_size.width;
				int yr = (r + 1) * gray_image_1.rows / grid_size.height;

				UMat gray_image_part = gray_image_1(Range(yl, yr), Range(xl, xr));
				orb->detectAndCompute(gray_image_part, UMat(), points, descriptors);
				keypoints_1.reserve(keypoints_1.size() + points.size());
				for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp)
				{
					kp->pt.x += xl;
					kp->pt.y += yl;
					keypoints_1.push_back(*kp);
				}
				_descriptors.push_back(descriptors.getMat(ACCESS_READ));
			}
		_descriptors.copyTo(descriptors_1);
		points.clear();
		_descriptors.release();
		descriptors.release();

		//��ȡ�ڶ���ͼƬ�������㼰����
		for (int r = 0; r < grid_size.height; ++r)
			for (int c = 0; c < grid_size.width; ++c)
			{
				int xl = c * gray_image_2.cols / grid_size.width;
				int yl = r * gray_image_2.rows / grid_size.height;
				int xr = (c + 1) * gray_image_2.cols / grid_size.width;
				int yr = (r + 1) * gray_image_2.rows / grid_size.height;

				UMat gray_image_part = gray_image_2(Range(yl, yr), Range(xl, xr));
				orb->detectAndCompute(gray_image_part, UMat(), points, descriptors);
				keypoints_2.reserve(keypoints_2.size() + points.size());
				for (std::vector<KeyPoint>::iterator kp = points.begin(); kp != points.end(); ++kp)
				{
					kp->pt.x += xl;
					kp->pt.y += yl;
					keypoints_2.push_back(*kp);
				}
				_descriptors.push_back(descriptors.getMat(ACCESS_READ));
			}
		_descriptors.copyTo(descriptors_2);
		points.clear();
		_descriptors.release();
		descriptors.release();
		cout << "��������ȡ��ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;

		//������ƥ��
		cout << "������ƥ��" << endl;
		t = getTickCount();
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
		cout << "������ƥ���ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
		//Mat first_match;
		//drawMatches(images[n], keypoints_1, images[n+1], keypoints_2, matches_info.matches, first_match);
		//imshow("first_match ", first_match);
		//waitKey();


		//ͼ����׼
		cout << "ͼ����׼" << endl;
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

		//������׼ͼ���ĸ���������
		CalcCorners(matches_info.H, images[n + 1]);

		//ͼ����׼  
		Mat imageTransform1, imageTransform2;
		cv::warpPerspective(images[n + 1], imageTransform2, matches_info.H, Size(MAX(corners.right_top.x, corners.right_bottom.x), images[n].rows));
		//imshow("ֱ�Ӿ���͸�Ӿ���任", imageTransform2);
		//waitKey();
		//imwrite("trans_pic.jpg", imageTransform2);
		cout << "ͼ����׼��ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;


		//ͼ��ƴ��
		cout << "ͼ��ƴ��" << endl;
		t = getTickCount();
		//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
		int dst_width = imageTransform2.cols;
		int dst_height = images[n].rows;

		Mat dst(dst_height, dst_width, CV_8UC3);
		dst.setTo(0);

		imageTransform2.copyTo(dst(Rect(0, 0, imageTransform2.cols, imageTransform2.rows)));
		images[n].copyTo(dst(Rect(0, 0, images[n].cols, images[n].rows)));
		OptimizeSeam(images[n], imageTransform2, dst);
		string file_name = "dst" + to_string(num) + ".jpg";
		imwrite(file_name, dst);
		cout << "ͼ��ƴ�Ӻ�ʱ: " << ((getTickCount() - t) / getTickFrequency()) << " sec\n" << endl;
		cout << "�ܺ�ʱ: " << ((getTickCount() - app_start_time) / getTickFrequency()) << " sec\n\n" << endl;
	}
	return 0;
}

void CalcCorners(const Mat& H, const Mat& src)
{
	double v2[] = { 0, 0, 1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

	V1 = H * V2;
	//���Ͻ�(0,0,1)
	corners.left_top.x = v1[0] / v1[2];
	corners.left_top.y = v1[1] / v1[2];

	//���½�(0,src.rows,1)
	v2[0] = 0;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.left_bottom.x = v1[0] / v1[2];
	corners.left_bottom.y = v1[1] / v1[2];

	//���Ͻ�(src.cols,0,1)
	v2[0] = src.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_top.x = v1[0] / v1[2];
	corners.right_top.y = v1[1] / v1[2];

	//���½�(src.cols,src.rows,1)
	v2[0] = src.cols;
	v2[1] = src.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = H * V2;
	corners.right_bottom.x = v1[0] / v1[2];
	corners.right_bottom.y = v1[1] / v1[2];

}

//�Ż���ͼ�����Ӵ���ʹ��ƴ����Ȼ
void OptimizeSeam(Mat& img1, Mat& trans, Mat& dst)
{
	int start = MIN(corners.left_top.x, corners.left_bottom.x);//��ʼλ�ã����ص��������߽�  

	double processWidth = img1.cols - start;//�ص�����Ŀ��  
	int rows = dst.rows;
	int cols = img1.cols; //ע�⣬������*ͨ����
	double alpha = 1;//img1�����ص�Ȩ��  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = img1.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
		uchar* t = trans.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//�������ͼ��trans�������صĺڵ㣬����ȫ����img1�е�����
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ�������ȣ�ʵ��֤�������ַ���ȷʵ��  
				alpha = (processWidth - (j - start)) / processWidth;
			}

			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);

		}
	}

}