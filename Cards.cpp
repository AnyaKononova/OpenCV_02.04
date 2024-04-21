#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <vector>

using namespace std;
using namespace cv;

void collectCards(vector<Mat>& cardsImages, vector<string>& cardsNames, vector<Mat>& cardsDescriptors, vector<vector<KeyPoint>>& cardsKeypoints)
{
	vector<string> imagePaths = {
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/tuz_love.jpg",
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/8_bub.jpg",
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/6_vin.jpg",
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/tuz_cross.jpg",
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/king_bub.jpg",
		"C:/Users/Nuta/Documents/Open CV/Практика 02.04/king_love.jpg"
	};

	Ptr<ORB> detector = ORB::create();
	for (const auto& imagePath : imagePaths) {
		Mat card = imread(imagePath);
		if (card.empty()) {
			cerr << "Error " << imagePath << endl;
			continue;
		}
		cardsImages.push_back(card);
		string cardName = imagePath.substr(imagePath.find_last_of('/') + 1);
		cardName = cardName.substr(0, cardName.find_last_of('.'));
		cardsNames.push_back(cardName);

		vector<KeyPoint> cardKeypoints;
		Mat cardDescriptors;
		detector->detectAndCompute(card, noArray(), cardKeypoints, cardDescriptors);
		cardsKeypoints.push_back(cardKeypoints);
		cardsDescriptors.push_back(cardDescriptors);
	}
}

int main()
{
	vector<Mat> cardsImages;
	vector<string> cardsNames;
	vector<Mat> cardsDescriptors;
	vector<vector<KeyPoint>> keypoints;

	collectCards(cardsImages, cardsNames, cardsDescriptors, keypoints);

	const Mat input = imread("C:/Users/Nuta/Documents/Open CV/Практика 02.04/start.jpg");
	if (!input.data)
	{
		printf("Error!");
		return -1;
	}
	cv::Mat img, greyImg, image1, image2;
	cvtColor(input, greyImg, COLOR_BGR2GRAY);
	GaussianBlur(greyImg, image1, Size(3, 3), 0);
	imshow("Gauss", image1);
	imwrite("GaussianBlur.jpg", image1);
	threshold(image1, image2, 215, 255, THRESH_BINARY);
	imshow("threshold", image2);
	imwrite("threshold.jpg", image2);

	std::vector<std::vector<Point>> contours;
	std::vector<Vec4i> hierarchy;
	findContours(image2, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Ptr<ORB> detector = ORB::create();
	Ptr<BFMatcher> matcher = BFMatcher::create();

	for (const auto& contour : contours)
	{
		vector<Point> contoursPoly;
		approxPolyDP(contour, contoursPoly, 1, true);
		RotatedRect cardRect = minAreaRect(contoursPoly);

		Mat card, rotatedMatrix, rotatedImage;
		string cardName;

		rotatedMatrix = getRotationMatrix2D(cardRect.center, cardRect.angle, 1.0);
		warpAffine(input, rotatedImage, rotatedMatrix, input.size(), INTER_CUBIC);
		getRectSubPix(rotatedImage, cardRect.size, cardRect.center, card);

		rotate(card, card, ROTATE_180);

		if (card.size[0] < card.size[1])
		{
			rotate(card, card, ROTATE_90_CLOCKWISE);
		}

		Mat cardDescriptors;
		vector<KeyPoint> cardKeypoints;
		detector->detectAndCompute(card, noArray(), cardKeypoints, cardDescriptors);

		if (cardDescriptors.empty()) {
			cardName = "";
		}
		else {
			int maxI = -1;
			int maxCount = 0;

			for (int i = 0; i < cardsImages.size(); i++) {
				if (cardsDescriptors[i].empty()) {
					continue;
				}

				vector<vector<DMatch>> knn_matches;
				matcher->knnMatch(cardsDescriptors[i], cardDescriptors, knn_matches, 3);
				vector<DMatch> correct;

				for (size_t i = 0; i < knn_matches.size(); i++) {
					if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
						correct.push_back(knn_matches[i][0]);
					}
				}

				if (maxCount < correct.size()) {
					maxCount = static_cast<int>(correct.size());
					maxI = i;
				}
			}

			if (maxI == -1) {
				cardName = "";
			}
			else {
				cardName = cardsNames[maxI];
			}
		}

		if (cardName != "")
		{
			Point2f boxPoints[4];
			cardRect.points(boxPoints);

			for (int j = 0; j < 4; j++)
			{
				line(input, boxPoints[j], boxPoints[(j + 1) % 4], Scalar(255, 0, 0), 5, LINE_AA);
			}
			putText(input, cardName, (boxPoints[0] + boxPoints[1]) * 0.5, FONT_HERSHEY_DUPLEX, 1.5, Scalar(0, 0, 255), 2, LINE_AA);
		}
	}

	imshow("Finish", input);
	imwrite("Finish.jpg", input);
	waitKey(0);

	return 0;
}


