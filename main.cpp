#include <opencv2/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video.hpp>
#include <iostream>
#include <conio.h>
#include <Windows.h>
#include <stdio.h>
#include "faceMemory.h";

#pragma warning(disable:4996)
#define CVUI_IMPLEMENTATION
#include "cvui.h"

#include <string.h>
#include <shobjidl.h>
#include <tchar.h>

using namespace std;
using namespace cv;

String path = "C:/opencv/sources/data/haarcascades/";
String
face_cascade_name = String(path + "haarcascade_frontalface_alt.xml"), //odpowiada za wykrywanie twarzy
eyes_cascade_name = String(path + "haarcascade_eye_tree_eyeglasses.xml"); //odpowiada za wykrywanie oczu
CascadeClassifier face_cascade, eyes_cascade;
int buttn;
int number = 2;
Mat face0;
VideoCapture cap;
//deklaracja zmiennych
int FactorOfScale = 15, min_FactorOfScale = 15, max_FactorOfScale = 100;
int AmountOfFrames = 30, min_AmountOfFrames = 20, max_AmountOfFrames = 200;
int minNumberOfNeighbors = 3, minNumberOfNeighbors_min = 3, minNumberOfNeighbors_max = 10;
int minSize = 100, minSize_min = 100, minSize_max = 450;
bool openCamera = false, openFilm = false, circleOption = false, blackEyes = false, bluredFace = false, faceSwap = false;
bool recordOption = false, finishRecord = false, turnOn = false, addPhoto = false, faceSwapWithPhoto = false, faceSwapWithLivePhoto = false;
double newFactorOfScale = (double)FactorOfScale / max_FactorOfScale * 10;
Size newMinimum_Size = Size(minSize, minSize);
vector<faceMemory> facesList;
string filePath = "";
string getTime(bool frame);
void saveVideo(vector<Mat> images, int width, int height);
Point getCenter(Rect rect);
int getDistance(Point p1, Point p2);
int menu(int number, string filePath);
string openfilename();

int main(void)
{
	Mat FrameOfControler = cv::Mat(640, 720, CV_8UC3); //skalowanie rozdzielczości, parametry w nawiasie
	string WINDOW_NAME = "Controller Frame";
	cvui::init(WINDOW_NAME);

	while (true) //odpowiada za pierwsze okienko i wybor opcji lub wyjście z programu
	{
		FrameOfControler = cv::Scalar(49, 52, 49);
		cvui::text(FrameOfControler, (380 / 2 - 140), 60, "CHANGE", 2.5, 0xcc0099);
		cvui::text(FrameOfControler, (380 / 2 - 140), 130, "YOUR FACE ", 2.5, 0xcc0099);

		if (cvui::button(FrameOfControler, (380 / 2 - 130), 255, "Open camera")) //otwarcie kamery internetowej
		{
			menu(number = 1, "");
			turnOn = true;
			break;
		}
		if (cvui::button(FrameOfControler, (380 / 2 + 90), 255, "Load video")) //otwarcie pliku video
		{
			menu(number = 2, openfilename().c_str());
			turnOn = true;
			break;
		}

		cv::imshow(WINDOW_NAME, FrameOfControler);

		if (cv::waitKey(20) == 27)
			break;
	}

	if (!turnOn) return 0;
	namedWindow("Frame", WINDOW_AUTOSIZE);
	Mat frame;
	cvui::init(WINDOW_NAME);

	if (!face_cascade.load(face_cascade_name) || !eyes_cascade.load(eyes_cascade_name)) //sprawdzenie czy plik moze się zaladowac
	{
		printf("Error while reading the file for detection \n"); return -1;
	}

	vector<Mat> images_Record; //wykrywanie twarzy ze zdjecia
	int counterFrames = 0; // liczenie klatek, dopoki jest doczyt klatek, to jest zmiana rozmiaru
	string facePhotoName;
	Mat facePhoto, faceToSwapLive;
	while (cap.read(frame)) {

		resize(frame, frame, Size(680, 480), 0.5, 0.5);
		std::vector<Rect> faces; // zapisywanie klatki z twarza
		Mat frame_gray;

		// czyszczenie ramki
		FrameOfControler = Scalar(49, 52, 49);

		if (cvui::button(FrameOfControler, 250, 40, "Add photo to change"))
		{
			addPhoto = true;
			facePhoto = imread(openfilename().c_str());
		}
		// ustawianie skali
		int  TrackbarField = 60, TextField = 20, LastPosition = 20;
		cvui::text(FrameOfControler, 20, LastPosition += TextField, "Scaling factor", 0.4, 0xcc0099);
		cvui::trackbar(FrameOfControler, 20, LastPosition += TextField, 150, &FactorOfScale, min_FactorOfScale, max_FactorOfScale);
		cvui::text(FrameOfControler, 20, LastPosition += TrackbarField, "Minimum number of neighbors", 0.4, 0xcc0099);
		cvui::trackbar(FrameOfControler, 20, LastPosition += TextField, 150, &minNumberOfNeighbors, minNumberOfNeighbors_min, minNumberOfNeighbors_max);
		cvui::text(FrameOfControler, 20, LastPosition += TrackbarField, "Minimum size", 0.4, 0xcc0099);
		cvui::trackbar(FrameOfControler, 20, LastPosition += TextField, 150, &minSize, minSize_min, minSize_max);
		cvui::text(FrameOfControler, 20, LastPosition += TrackbarField, "Amount of frames to save", 0.4, 0xcc0099);
		cvui::trackbar(FrameOfControler, 20, LastPosition += TextField, 150, &AmountOfFrames, min_AmountOfFrames, max_AmountOfFrames);

		// ustawianie opcji zakreslania twarzy, zaslaniania oczu, rozmycia twarzy
		TrackbarField = 60, TextField = 40, LastPosition = 40;
		cvui::checkbox(FrameOfControler, 250, LastPosition += TrackbarField, "Detect Face", &circleOption);
		cvui::checkbox(FrameOfControler, 250, LastPosition += TextField, "Eyes Covering", &blackEyes);
		cvui::checkbox(FrameOfControler, 250, LastPosition += TextField, "Blured Face", &bluredFace);
		cvui::checkbox(FrameOfControler, 250, LastPosition += TextField, "Face Swap Option", &faceSwap);
		cvui::checkbox(FrameOfControler, 250, LastPosition += TextField, "Face Swap With Photo", &faceSwapWithPhoto);
		cvui::text(FrameOfControler, 250, LastPosition += TextField, "Pick one to start recording", 0.4, 0xcc0099);
		// zapis 
		LastPosition += TextField;
		for (int i = 0; i < facesList.size(); i++)
		{
			cv::Mat out = facesList[i].image.clone();
			resize(out, out, Size(40, 40));
			if (!out.empty())
				if (facesList[i].isRecorded)
					rectangle(FrameOfControler, Point(-3 + 250 + i * 50, -3 + LastPosition), Point(3 + 40 + 250 + i * 50, 3 + 40 + LastPosition), Scalar(255, 0, 0), 2);

			if (cvui::button(FrameOfControler, 250 + i * 50, LastPosition, out, out, out)) {
				facesList[i].isRecorded = !facesList[i].isRecorded;
			}
		}


		cvui::text(FrameOfControler, 250, LastPosition += TextField + 20, "Pick one to start swapping", 0.4, 0xcc0099);
		LastPosition += TextField;
		for (int i = 0; i < facesList.size(); i++)
		{
			cv::Mat out = facesList[i].image.clone();
			resize(out, out, Size(40, 40));
			if (!out.empty())
				if (facesList[i].isToSwap)
					rectangle(FrameOfControler, Point(-3 + 250 + i * 50, -3 + LastPosition), Point(3 + 40 + 250 + i * 50, 3 + 40 + LastPosition), Scalar(255, 0, 0), 2); //rysowanie prostokata zakrywajacego oczy, na koncu sa wspolrzedne RGB-koloru

			if (cvui::button(FrameOfControler, 250 + i * 50, LastPosition, out, out, out)) {
				if (faceSwapWithLivePhoto)
					facesList[i].isToSwap = !facesList[i].isToSwap;
			}
		}
		cvui::checkbox(FrameOfControler, 250, LastPosition += TextField + 10, "Face Swap Option With Live", &faceSwapWithLivePhoto);

		cvui::update();
		cv::imshow(WINDOW_NAME, FrameOfControler);

		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		Mat frame_face_copy = frame.clone();


		face_cascade.detectMultiScale(frame_gray, faces, newFactorOfScale, minNumberOfNeighbors, 0, newMinimum_Size);

		for (int i = 0; i < facesList.size(); i++)
			++facesList[i].counter;

		for (int i = 0; i < faces.size(); i++)
		{
			bool faceExists = false;
			Point centerPresentFace = getCenter(faces[i]);
			for (faceMemory &faceMem : facesList)
			{
				//umieszczenie kwadratu zasłaniającego oczy
				if (getDistance(faceMem.getCenter(), centerPresentFace) <= 100) {
					faceMem.rect = faces[i];
					faceMem.image = frame(faces[i]).clone();
					faceMem.counter = 0;
					faceExists = true;
					break;
				}
			}
			if (!faceExists)
				facesList.push_back(faceMemory(faces[i], frame(faces[i])));
		}

		vector<faceMemory> usableFaces;
		for (int i = 0; i < facesList.size(); i++)
		{
			bool toDelete = facesList[i].counter == 30;
			if (facesList[i].isRecorded)
				facesList[i].recordedImages.push_back(facesList[i].image);
			if ((facesList[i].recordedImages.size() >= AmountOfFrames) || (toDelete && facesList[i].isRecorded)) {
				facesList[i].isRecorded = false;
				saveVideo(facesList[i].recordedImages, facesList[i].rect.width, facesList[i].rect.height);
				facesList[i].recordedImages.clear();
			}
			if (facesList[i].counter == 0)
				usableFaces.push_back(facesList[i]);
			if (facesList[i].isToSwap)
				faceToSwapLive = facesList[i].image.clone();
			if (toDelete)
				facesList.erase(facesList.begin() + i);
		}

		Mat frame_out = frame.clone();
		//funkcje odpowiedzialne za zaznaczanie twarzy, zakrycie oczu, zamazanie twarzy
		for (int i = 0; i < usableFaces.size(); i++)
		{

			if (bluredFace)
				GaussianBlur(frame(usableFaces[i].rect), frame_out(usableFaces[i].rect), Size(15, 15), 20); // zamazanie twarzy (znieksztalcenie Gaussa)
			if (circleOption) // zakreslanie twarzy okregiem
				circle(frame_out, Point(usableFaces[i].rect.x + usableFaces[i].rect.width / 2, usableFaces[i].rect.y + usableFaces[i].rect.height / 2), cvRound((usableFaces[i].rect.width + usableFaces[i].rect.height)*0.25), Scalar(0, 100, 0), 3);
			if (blackEyes)
				rectangle(frame_out, Rect(Point(usableFaces[i].rect.x, usableFaces[i].rect.y + usableFaces[i].rect.height / 2) - Point(0, usableFaces[i].rect.height / 3), Size(usableFaces[i].rect.width, usableFaces[i].rect.height / 3)), Scalar(0, 0, 0), -1);
			if (faceSwap && (usableFaces.size() >= 2)) {
				Mat Resize_Face = frame(usableFaces[i].rect).clone();
				resize(Resize_Face, Resize_Face, usableFaces[(i + 1) % usableFaces.size()].rect.size());
				Resize_Face.copyTo(frame_out(usableFaces[(i + 1) % usableFaces.size()].rect));
			}
			if (faceSwapWithPhoto && !facePhoto.empty()) //jezeli wyktywana jest twarz, to zamien ja
			{
				Mat Resize_Face = facePhoto.clone();
				resize(Resize_Face, Resize_Face, usableFaces[i].rect.size());
				Resize_Face.copyTo(frame_out(usableFaces[i].rect));
			}
			if (faceSwapWithLivePhoto && !faceToSwapLive.empty()) // zamiana twarzy
			{
				Mat Resize_Face = faceToSwapLive.clone();
				resize(Resize_Face, Resize_Face, usableFaces[i].rect.size());
				Resize_Face.copyTo(frame_out(usableFaces[i].rect));
			}
		}

		imshow("Frame", frame_out);
		buttn = waitKey(1);

		if (buttn == 114)  // 'r'
		{
			recordOption = !recordOption;
			if (recordOption) {
				counterFrames = 0;
				images_Record.clear();
			}
			else {
				finishRecord = true;
			}
		}

		if (recordOption || finishRecord)
		{
			++counterFrames;
			images_Record.push_back(frame.clone());
			cout << "is recording" << endl;

			if (counterFrames >= AmountOfFrames || finishRecord)
			{
				cout << images_Record.size() << endl;
				saveVideo(images_Record, frame.size().width, frame.size().height);
				recordOption = false;
				finishRecord = false;
			}
		}
		if (buttn == 27)
			break;
	}
	cap.release();
	destroyAllWindows();
	return 0;
}

// obsługa okienka wyboru pliku
string openfilename() {
	string fileName = "";
	OPENFILENAME open_file_name, type; //sugestia wyboru pliku
	char szFileName[MAX_PATH] = "";
	ZeroMemory(&open_file_name, sizeof(open_file_name));

	open_file_name.lStructSize = sizeof(OPENFILENAME);
	open_file_name.hwndOwner = NULL;
	if (number == 2)
		open_file_name.lpstrFilter = _T("Video\0*.avi\0");
	if (addPhoto)
		open_file_name.lpstrFilter = _T("JPG\0*.jpg\0");
	open_file_name.lpstrFile = szFileName;
	open_file_name.nMaxFile = MAX_PATH;
	open_file_name.Flags = OFN_EXPLORER | OFN_FILEMUSTEXIST | OFN_HIDEREADONLY;

	if (GetOpenFileName(&open_file_name))
		fileName.assign(szFileName);

	ZeroMemory(&open_file_name, sizeof(open_file_name));
	return fileName;
}

int menu(int number, string filePath)
{
	if (openCamera)
		number = 1;
	if (openFilm)
		number = 2;

	switch (number) {
	case 1:
		cap.open(0);
		break;
	case 2:
		cap.open(filePath);
		break;
	default:
		cout << "could not be opened";
		return -1;
	}
}
string getTime(bool frame = true)
{
	time_t timeObj;
	time(&timeObj);
	tm *pTime = gmtime(&timeObj);
	char buffer[100];
	sprintf(buffer, "%d-%d-%d-%d-%d-%d", pTime->tm_year + 1900, pTime->tm_mon + 1, pTime->tm_mday, pTime->tm_hour, pTime->tm_min, pTime->tm_sec);
	return buffer;
}
// zapis wideo
void saveVideo(vector<Mat> images, int width, int height)
{
	VideoWriter video("face record - " + getTime(false) + ".avi", VideoWriter::fourcc('D', 'I', 'V', '3'), 27, Size(width, height));
	for (size_t i = 0; i < images.size(); i++)
	{
		resize(images[i], images[i], Size(width, height));
		video.write(images[i]);
	}
	string response = !video.isOpened() ? "Output video could not be opened" : "video saved";
	cout << response << endl;
}
//obliczanie środka na podstawie kwadratu
Point getCenter(Rect rect) {
	return Point(rect.x + (rect.width / 2), rect.y + (rect.height / 2));
}

int getDistance(Point p1, Point p2) {

	return sqrt(pow(p1.x - p2.x, 2) + pow(p1.y - p2.y, 2));
}