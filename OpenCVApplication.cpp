// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "Functions.h"

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // no dword alignment is done !!!
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
				/* sau puteti scrie:
				uchar val = lpSrc[i*width + j];
				lpDst[i*width + j] = 255 - val;
				//	w = width pt. imagini cu 8 biti / pixel
				//	w = 3*width pt. imagini cu 24 biti / pixel
				*/
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step; // latimea in octeti a unei linii de imagine
		
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);
		
		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* dstDataPtrH = dstH.data;
		uchar* dstDataPtrS = dstS.data;
		uchar* dstDataPtrV = dstV.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				// sau int hi = i*w + j * 3;	//w = 3*width pt. imagini 24 biti/pixel
				int gi = i*width + j;
				
				dstDataPtrH[gi] = hsvDataPtr[hi] * 510/360;		// H = 0 .. 255
				dstDataPtrS[gi] = hsvDataPtr[hi + 1];			// S = 0 .. 255
				dstDataPtrV[gi] = hsvDataPtr[hi + 2];			// V = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);
		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey();
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey();  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/// <summary>
/// Labor 1
/// </summary>


void hsvCoordinates()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat sursa = imread(fname);
		Mat hsvImg;
		cvtColor(sursa, hsvImg, CV_BGR2HSV);


//		Mat dst = testBGR2HSVVersion2(sursa);

		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &hsvImg);

		imshow("sursa", sursa);
		imshow("My Window", hsvImg);

		waitKey();
	}
}
Mat srcMat;
void MyCallBackFuncMouse(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;

	if (event == CV_EVENT_MOUSEMOVE)
	{
		Mat cpy = srcMat.clone();
		char str[100];
		printf("Pos(x,y): %d,%d  Color(HSV): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		sprintf(str, "Pos(x,y): %d,%d  Color(HSV): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);

		putText(cpy, str , Point(5,20), 2, 0.4, Vec3b(0,0,255), 1, 8);
		imshow("Cpy", cpy);
	}
}


void putTextHSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		srcMat = imread(fname);
		Mat hsvMouse;
		cvtColor(srcMat, hsvMouse, CV_BGR2HSV);

		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFuncMouse, &hsvMouse);

		imshow("My Window", hsvMouse);

		waitKey();
	}
}

/// <summary>
/// Labor 2
/// </summary>

int* histogram(Mat src)
{
	int* histogram = (int*)calloc(255, sizeof(int));
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			histogram[src.at<uchar>(i, j)]++;
		}
	}
	return histogram;
}

int treshold = 0;
void OtsuAlgorithm(Mat src)
{
	Mat dst = src.clone();
	int N = src.rows * src.cols;
	int max_intensity = 255;
	treshold = 0;
	float var_max = 0, sumT = 0, sumC1 = 0, q1 = 0, q2 = 0, mi1 = 0, mi2 = 0;
	int* hist = histogram(src);
	float* betweenClassVariance = (float*)calloc(256, sizeof(float));

	for (int i = 0; i <= max_intensity; i++)
	{
		sumT += i * hist[i];
	}

	for (int t = 0; t <= max_intensity; t++)
	{
		q1 += hist[t];
		if (q1 == 0)
			continue;
		q2 = N - q1;

		sumC1 += t * hist[t];
		mi1 = sumC1 / q1;
		mi2 = (sumT - sumC1) / q2;

		betweenClassVariance[t] = q1 * q2 * (mi1 - mi2) * (mi1 - mi2);

		if (betweenClassVariance[t] > var_max)
		{
			treshold = t;
			var_max = betweenClassVariance[t];
		}
	}
	printf("\nOtsu treshold = %d\n", treshold);
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<uchar>(i, j) > treshold)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	imshow("DST", dst);
	waitKey(0);
}

void conversionRGBtoHSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;
		int w = src.step;

		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		Mat splited[3];
		split(hsvImg, splited);
		dstH = splited[0];
		dstS = splited[1];
		dstV = splited[2];

		dstH = dstH * 255 / 180;
		imshow("Src", src);
		imshow("H", dstH);
		imshow("S", dstS);
		imshow("V", dstV);

		int* histoH = histogram(dstH);
		int* histoS = histogram(dstS);
		int* histoV = histogram(dstV);

		showHistogram("histH", histoH, 255, 200, true);
		showHistogram("histS", histoS, 255, 200, true);
		showHistogram("histV", histoV, 255, 200, true);


		OtsuAlgorithm(dstH);
	}
}


/// <summary>
/// Labor 3
/// </summary>
/// <returns></returns>


#define MAX_HUE 256
int histc_hue[MAX_HUE];
Point Pstart, Pend;
float Hmean = 16.0, Hstd = 5.0;

void L3_ColorModel_Init()
{
	memset(histc_hue, 0, sizeof(unsigned int) * MAX_HUE);
}

void CallBackFuncL3(int event, int x, int y, int flags, void* userdata)
{
	Mat* H = (Mat*)userdata;
	Rect roi; // regiunea de interes curenta (ROI)
	if (event == EVENT_LBUTTONDOWN)
	{
		// punctul de start al ROI
		Pstart.x = x;
		Pstart.y = y;
		printf("Pstart: (%d, %d) ", Pstart.x, Pstart.y);
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		// punctul de final (diametral opus) al ROI
		Pend.x = x;
		Pend.y = y;
		printf("Pend: (%d, %d) ", Pend.x, Pend.y);
		// sortare puncte dupa x si y
		//(parametrii width si height ai structurii Rect > 0)
		roi.x = min(Pstart.x, Pend.x);
		roi.y = min(Pstart.y, Pend.y);
		roi.width = abs(Pstart.x - Pend.x);
		roi.height = abs(Pstart.y - Pend.y);
		printf("Local ROI: (%d, %d), (%d, %d)\n", roi.x, roi.y, roi.x + roi.width,
			roi.y + roi.height);
		int hist_hue[MAX_HUE]; // histograma locala a lui Hue
		memset(hist_hue, 0, MAX_HUE * sizeof(int));
		// Din toata imaginea H se selecteaza o subimagine (Hroi) aferenta ROI
		Mat Hroi = (*H)(roi);
		uchar hue;
		//construieste histograma locala aferente ROI
		for (int y = 0; y < roi.height; y++)
			for (int x = 0; x < roi.width; x++)
			{
				hue = Hroi.at<uchar>(y, x);
				hist_hue[hue]++;
			}
		//acumuleaza histograma locala in cea globala/cumulativa
		for (int i = 0; i < MAX_HUE; i++)
			histc_hue[i] += hist_hue[i];
		// afiseaza histohrama locala
		showHistogram("H local histogram", hist_hue, MAX_HUE, 200, true);
		// afiseaza histohrama globala / cumulativa
		showHistogram("H global histogram", histc_hue, MAX_HUE, 200, true);
	}
}

void L3_ColorModel_Build()
{
	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		GaussianBlur(src, src, Size(5, 5), 0, 0);
		namedWindow("src", 1);
		Mat H = Mat(height, width, CV_8UC1);
		uchar* lpH = H.data;
		cvtColor(src, hsv, CV_BGR2HSV);

		uchar* hsvDataPtr = hsv.data;

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				int hi = i * width * 3 + j * 3;
				int gi = i * width + j;
				lpH[gi] = hsvDataPtr[hi] * 510 / 360;
			}
		}

		setMouseCallback("src", CallBackFuncL3, &H);
		imshow("src", src);
		waitKey(0);
	}
}

void L3_ColorModel_Save()
{
	int hue, sat, i, j;
	int histF_hue[MAX_HUE]; // histograma filtrata cu FTJ
	memset(histF_hue, 0, MAX_HUE * sizeof(unsigned int));
	//Filtrare histograma Hue (optional)
#define FILTER_HISTOGRAM 1
#if FILTER_HISTOGRAM == 1
// filtrare histograma cu filtru gaussian 1D de dimensiune w=7
	float gauss[7];
	float sqrt2pi = sqrtf(2 * PI);
	float sigma = 1.5;
	float e = 2.718;
	float sum = 0;
	// Construire gaussian
	for (i = 0; i < 7; i++) {
		gauss[i] = 1.0 / (sqrt2pi * sigma) * powf(e, -(float)(i - 3) * (i - 3)
			/ (2 * sigma * sigma));
		sum += gauss[i];
	}
	// Filtrare cu gaussian
	for (j = 3; j < MAX_HUE - 3; j++)
	{
		for (i = 0; i < 7; i++)
			histF_hue[j] += (float)histc_hue[j + i - 3] * gauss[i];
	}
#elif
	for (j = 0; j < MAX_HUE; j++)
		histF_hue[j] = histc_hue[j];
#endif // End of "Filtrare Gaussiana Histograma Hue"

	// pregatire pt. scriere valoari model in fisier
	FILE* fp;
	// Hue
	fp = fopen("C:\\Users\\CatalinBriscan\\source\\repos\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\OpenCVApplication-VS2019_OCV3411_basic_IOM\\test.txt", "wt");
	fprintf(fp, "H=[\n");
	for (hue = 0; hue < MAX_HUE; hue++) {
		fprintf(fp, "%d\n", histF_hue[hue]);
	}
	fprintf(fp, "];\n");
	fprintf(fp, "Hmean = %.0f ;\n", Hmean);
	fprintf(fp, "Hstd = %.0f ;\n", Hstd);
	fclose(fp);


	showHistogram("H global histogram", histc_hue, MAX_HUE, 200, true);
	showHistogram("H global filtered histogram", histF_hue, MAX_HUE, 200, true);
	// Wait until user press some key
	waitKey(0);
} //end of L3_ColorModel_Save()

int Arie(Mat src, Vec3b p)
{
	int arie = 0;
	for (int i = 0; i < src.rows; i++)
	{
		for (int j = 0; j < src.cols; j++)
		{
			if (src.at<Vec3b>(i, j)[2] == p[2] &&
				src.at<Vec3b>(i, j)[1] == p[1] &&
				src.at<Vec3b>(i, j)[0] == p[0])
				arie++;
		}
	}
	return arie;
}

void CentruDeMasa(Mat src, int arie, Vec3b p, float* rb, float* cb)
{
	int sumr = 0, sumc = 0;

	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			if (src.at<Vec3b>(r, c)[2] == p[2] &&
				src.at<Vec3b>(r, c)[1] == p[1] &&
				src.at<Vec3b>(r, c)[0] == p[0])
			{
				sumr += r;
				sumc += c;
			}
		}
	}
	*rb = (1.0 / arie) * sumr;
	*cb = (1.0 / arie) * sumc;
}

void AxaAlungire(Mat src, float* rb, float* cb, long double* tangenta, Vec3b p)
{
	long double suma_numarator = 0, suma_numitor1 = 0, suma_numitor2 = 0;

	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			if (src.at<Vec3b>(r, c)[2] == p[2] &&
				src.at<Vec3b>(r, c)[1] == p[1] &&
				src.at<Vec3b>(r, c)[0] == p[0])
			{
				suma_numarator += (r - *rb) * (c - *cb);
				suma_numitor1 += (c - *cb) * (c - *cb);
				suma_numitor2 += (r - *rb) * (r - *rb);
			}
		}
	}

	suma_numarator *= 2;
	int suma_numitor = suma_numitor1 - suma_numitor2;
	*tangenta = atan2(suma_numarator, suma_numitor) / 2;
}

void onMouse(int event, int x, int y, int flags, void* param)
{
	Mat src = *(Mat*)param;

	if (event == EVENT_LBUTTONDOWN)
	{
		if (src.at<Vec3b>(y, x)[2] == 255 &&
			src.at<Vec3b>(y, x)[1] == 255 &&
			src.at<Vec3b>(y, x)[0] == 255)
			return;
		else
		{
			int arie = Arie(src, src.at<Vec3b>(y, x));
			printf("\nAria este : %d", arie);

			float rb = 0.0, cb = 0.0;
			CentruDeMasa(src, arie, src.at<Vec3b>(y, x), &rb, &cb);
			printf("\nCentrul de masa este : (%.2f, %.2f)", rb, cb);

			long double axa_alungire = 0;
			AxaAlungire(src, &rb, &cb, &axa_alungire, src.at<Vec3b>(y, x));
			printf("\nAxa de alungire este : %.2Lf", axa_alungire * 180.0 / PI);


			float xval, yval;
			xval = (-cb) / axa_alungire + rb;
			yval = axa_alungire * rb + cb;

			line(src, Point(0, (int)(xval)),
				Point((int)(rb), (int)(cb + yval)), Scalar(255, 255, 0));
			imshow("Dst", src);
		}
	}
}

double xcpr, ycpr;
float teta;
void Labeling1(const string& name, const Mat& src, bool output_format, double* xc, double* yc)
{
	// dst - matrice RGB24 pt. afisarea rezultatului
	Mat dst = Mat::zeros(src.size(), CV_8UC3);

	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;

	// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=findcontours#findcontours
	findContours(src, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE);

	Moments m;
	if (contours.size() > 0)
	{
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		int idx = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			const vector<Point>& c = contours[idx];

			// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=moments#moments
			m = moments(c); // calcul momente imagine
			double arie = m.m00; // aria componentei conexe idx

			if (arie > 0)
			{
				xcpr = m.m10 / m.m00; // coordonata x a CM al componentei conexe idx
				ycpr = m.m01 / m.m00; // coordonata y a CM al componentei conexe idx

				Scalar color(rand() & 255, rand() & 255, rand() & 255);

				// http://docs.opencv.org/2.4/modules/imgproc/doc/structural_analysis_and_shape_descriptors.html?highlight=drawcontours#drawcontours
				if (output_format) // desenare obiecte pline ~ etichetare
					drawContours(dst, contours, idx, color, CV_FILLED, 8, hierarchy);
				else  //desenare contur obiecte
					drawContours(dst, contours, idx, color, 1, 8, hierarchy);

				Point center(xcpr, ycpr);
				int radius = 5;

				// afisarea unor cercuri in jurul centrelor de masa
				//circle(final, center, radius,Scalar(255,255,355), 1, 8, 0);

				// afisarea unor cruci peste centrele de masa
				DrawCross(dst, center, 9, Scalar(255, 255, 255), 1);

				// https://en.wikipedia.org/wiki/Image_moment
				//calcul axa de alungire folosind momentele centarte de ordin 2
				double mc20p = m.m20 / m.m00 - xcpr * xcpr; // double mc20p = m.mu20 / m.m00;
				double mc02p = m.m02 / m.m00 - ycpr * ycpr; // double mc02p = m.mu02 / m.m00;
				double mc11p = m.m11 / m.m00 - xcpr * xcpr; // double mc11p = m.mu11 / m.m00;
				teta = 0.5 * atan2(2 * mc11p, mc20p - mc02p);
				float teta_deg = teta * 180 / PI;

				printf("ID=%d, arie=%.0f, xc=%0.f, yc=%0.f, teta=%.0f\n", idx, arie, xcpr, ycpr, teta_deg);

			}
		}
	}
}

void L3_ColorModel_Build_Prb1()
{
	Mat src;
	Mat hsv;
	float Hue_mean = 16.0;
	float Hue_std = 5.0;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		GaussianBlur(src, src, Size(5, 5), 0, 0);

		cvtColor(src, hsv, CV_BGR2HSV);

		Mat channels[3];
		split(hsv, channels);
		Mat dstH = channels[0];
		dstH = dstH * 255 / 180;

		float k = 2.5, minVal, maxVal;
		minVal = Hue_mean - k * Hue_std;
		maxVal = Hue_mean + k * Hue_std;

		Mat dst = Mat(height, width, CV_8UC1);
		Mat dst1 = Mat(height, width, CV_8UC3);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				if (dstH.at<uchar>(i, j) >= minVal && dstH.at<uchar>(i, j) <= maxVal)
				{
					dst.at<uchar>(i, j) = 0;
					dst1.at<Vec3b>(i, j) = { 0, 0, 0 };
				}
				else
				{
					dst.at<uchar>(i, j) = 255;
					dst1.at<Vec3b>(i, j) = { 255, 255, 255 };
				}
			}
		}

		imshow("src", src); 
		imshow("Segmentation", dst);

		// creare element structural de dimensiune 3x3 de tip patrat (V8)
		Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		// dilatare cu acest element structural (aplicata 2x)
		dilate(dst, dst, element2, Point(-1, -1), 2);
		erode(dst, dst, element2, Point(-1, -1), 4);
		dilate(dst, dst, element2, Point(-1, -1), 2);

		imshow("Postprocessing", dst);


		Labeling1("contur", dst, false, &xcpr, &ycpr);
		printf("xc = % 0.f, yc = % 0.f, teta = % .0f\n", xcpr, ycpr, teta);

		double slope = tan(teta);
		printf("%f ", slope);
		int y1 = 0;
		int x1 = (-ycpr) / slope + xcpr;

		int y2 = src.rows - 1;
		int x2 = (y2 - ycpr) / slope + xcpr;

		line(dst, Point(x1, y1), Point(x2, y2), CV_RGB(255, 0, 0), 1, 8, 0);
		imshow("Img", dst);
		
		waitKey(0);
		destroyAllWindows();
	}
}


void MyCallBackFunc4(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		Point clicked;
		clicked.x = x; 
		clicked.y = y;

		queue<Point> que;
		int N = 1, k = 1;
		que.push(clicked);
		Mat* src = (Mat*)param;
		Mat labels = Mat::zeros((*src).size(), CV_8UC1);
		labels.at<uchar>(y, x) = k;

		float hue_avg = (*src).at<uchar>(y,x);
		float T = Hstd * 2.5;

		while (!que.empty())
		{
			Point oldest = que.front();
			que.pop();
			int xcpy = oldest.x;
			int ycpy = oldest.y;
			for (int i = ycpy - 1; i <= ycpy + 1; i++)
			{
				for (int j = xcpy - 1; j <= xcpy + 1; j++)
				{
					if ((i > 0 && i < (*src).rows) && (j > 0 && j < (*src).cols))
					{
						if (labels.at<uchar>(i, j) == 0 && abs((*src).at<uchar>(i, j) - hue_avg) < T)
						{
							que.push(Point(j, i));
							labels.at<uchar>(i, j) = k;
							hue_avg = (N * hue_avg + (*src).at<uchar>(i, j)) / (N + 1);
							N += 1;
						}
					}
				}
			}

		}
	
		Mat dst = Mat::zeros((*src).size(), CV_8UC1);
		for (int i = 0; i < labels.rows; i++)
		{
			for (int j = 0; j < labels.cols; j++)
			{
				if (labels.at<uchar>(i, j) != 0)
				{
					dst.at<uchar>(i, j) = 255;
				}
				else
					dst.at<uchar>(i, j) = 0;
			}
		}
		Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
		// dilatare cu acest element structural (aplicata 2x)
		dilate(dst, dst, element2, Point(-1, -1), 2);
		erode(dst, dst, element2, Point(-1, -1), 2);


		imshow("Postprocesare", dst);
	
	}
}

void regionGrowing(int x, int y, Mat src)
{
	Point clicked;
	clicked.x = x;
	clicked.y = y;

	queue<Point> que;
	int N = 1, k = 1;
	que.push(clicked);
	Mat labels = Mat::zeros(src.size(), CV_8UC1);
	labels.at<uchar>(y, x) = k;
	float hue_avg = src.at<uchar>(y, x);
	while (!que.empty())
	{
		Point oldest = que.front();
		que.pop();
		int xcpy = oldest.x;
		int ycpy = oldest.y;
		printf("%d -- %d\n", xcpy, ycpy);
		for (int i = ycpy - 1; i <= ycpy + 1; i++)
		{
			for (int j = xcpy - 1; j <= xcpy + 1; j++)
			{
				if ((i > 0 && i < src.cols) && (j > 0 && j < src.rows))
				{
					if (labels.at<uchar>(i, j) == 0 && abs(src.at<uchar>(i, j) - hue_avg) < 15)
					{
						que.push(Point(j, i));
						labels.at<uchar>(i, j) = k;
						hue_avg = (N * hue_avg + src.at<uchar>(i, j)) / (N + 1);
						N += 1;
					}
				}
			}
		}

	}

	Mat dst = src.clone();
	int T = Hstd * 2.5;
	for (int i = 0; i < labels.rows; i++)
	{
		for (int j = 0; j < labels.cols; j++)
		{
			if (labels.at<uchar>(i, j) == T)
				dst.at<uchar>(i, j) = 255;
			else
				dst.at<uchar>(i, j) = 0;
		}
	}
	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));
	dilate(dst, dst, element2, Point(-1, -1), 2);
	erode(dst, dst, element2, Point(-1, -1), 2);


	imshow("Postprocesare", dst);

}

void L4_RegionGrowing()
{
	Mat src;
	Mat hsv;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		GaussianBlur(src, src, Size(5, 5), 0.8, 0.8);
		namedWindow("src", 1);
		Mat dstH = Mat(height, width, CV_8UC1);
		Mat dstS = Mat(height, width, CV_8UC1);
		Mat dstV = Mat(height, width, CV_8UC1);

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);
		Mat channels[3];
		split(hsvImg, channels);
		dstH = channels[0];
		dstS = channels[1];
		dstV = channels[2];
		dstH = dstH * 255 / 180;
		imshow("H", dstH);

		setMouseCallback("src", MyCallBackFunc4, &dstH);
		imshow("src", src);
		waitKey(0);
	}
}

int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - HsvCoordinates\n");
		printf(" 11 - Put text hsv\n");
		printf(" 12 - conversionRGBtoHSV\n");
		printf(" 13 - lab3 Intro\n");
		printf(" 14 - lab3 Problems\n");
		printf(" 15 - lab4 Problems\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				hsvCoordinates();
				break;
			case 11:
				putTextHSV();
				break;
			case 12:
				conversionRGBtoHSV();
				break;
			case 13:
				L3_ColorModel_Init();
				L3_ColorModel_Build();
				L3_ColorModel_Save();
				break;
			case 14:
				L3_ColorModel_Build_Prb1();
				break;
			case 15:
				L4_RegionGrowing();
				break;
		}
	}
	while (op!=0);
	return 0;
}