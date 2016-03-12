package vision.computer.opencv_android;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by Manuel on 09/03/2016.
 */
public class Recognition {

    private static final int NEXT_IMAGE = 1;
    private static final int PREVIOUS_IMAGE = -1;
    private static final int CV_FILL_CONTOURS = -1;

    private static View view;
    private static String path;
    private static ArrayList<String> files;
    private static int nImageIndex;

    public Recognition(View view, String path) {
        this.view = view;
        files = directories(path);
        nImageIndex = 0;
    }

    public static Mat loadImage(int next) {
        if (android.os.Environment.getExternalStorageState().equals(
                android.os.Environment.MEDIA_MOUNTED)) {
            if (next == NEXT_IMAGE) {
                if (nImageIndex == files.size() - 1)
                    nImageIndex = 0;
                else
                    nImageIndex++;
            } else if (next == PREVIOUS_IMAGE) {
                if (nImageIndex == 0)
                    nImageIndex = files.size() - 1;
                else
                    nImageIndex--;
            }

            Mat image = Imgcodecs.imread(path + files.get(nImageIndex));
            Snackbar snackbar = Snackbar.make(view, "Image loaded: " + files.get(nImageIndex), Snackbar.LENGTH_LONG);
            snackbar.show();
            return image;
        } else return null;
    }

    private ArrayList<String> directories(String path) {
        File root = Environment.getExternalStorageDirectory();
        this.path = root.getAbsolutePath() + path;
        File folder = new File(this.path);
        File[] listOfFiles = folder.listFiles();
        ArrayList<String> files = new ArrayList<String>();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile())
                files.add(listOfFiles[i].getName());
        }
        return files;
    }

    public Mat regularTresholding(Mat input) {
        Mat dst = new Mat();
        Imgproc.threshold(input, dst, 127, 255, Imgproc.THRESH_BINARY_INV);
        return dst;
    }

    public Mat otsuThresholding(Mat input, boolean gaussian) {
        Mat dst = new Mat(input.size(), input.type());
        Mat gray = new Mat(input.size(), input.type());

        if (gaussian)
            Imgproc.GaussianBlur(input, input, new Size(5, 5), 0);

        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(gray, dst, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        Imgproc.cvtColor(dst, input, Imgproc.COLOR_GRAY2BGR);
        return input;
    }

    public Mat contours(Mat input, int gaussian) {
        Random r = new Random();
        Filters f = new Filters();
        Mat gray = new Mat(input.size(), input.type());
        Mat canny = new Mat(input.size(), input.type());
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();

        //Treshold the image --> less errors/noise
        input = otsuThresholding(input, false);
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        //Filter to detect borders
        Imgproc.Canny(gray, canny, 4, 8);
        //Blur image so borders are connected
        if (gaussian == 1)
            canny = f.gaussianSmooth(canny, 1);
        else if (gaussian == 0) {
            Mat structure = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(40, 40));
            Imgproc.morphologyEx(canny, canny, Imgproc.MORPH_CLOSE, structure);
            canny = f.poster(canny, 2, 1);
        }
        //Find contours and change color range to BGR
        Imgproc.findContours(canny, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.cvtColor(canny, input, Imgproc.COLOR_GRAY2BGR);

        //Print contours
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            Imgproc.drawContours(input, contours, contourIdx,
                    new Scalar(1 * r.nextInt(255 / (contourIdx + 1)) * (contourIdx + 1),
                            1 * r.nextInt(255 / (contourIdx + 1)) * (contourIdx + 1),
                            1 * r.nextInt(255 / (contourIdx + 1)) * (contourIdx + 1)),
                    CV_FILL_CONTOURS);
        }

        return input;
    }

    public void getDescriptors(List<MatOfPoint> contours) {
        for (int i = 0; i < contours.size(); i++) {
            MatOfPoint c = contours.get(i);
            Moments moments = Imgproc.moments(c);

            int centroideX = (int) (moments.get_m10() / moments.get_m00());
            int centroideY = (int) (moments.get_m01() / moments.get_m00());

            double areaM = moments.get_m00();

            double area = Imgproc.contourArea(c);

            MatOfPoint2f c2 = new MatOfPoint2f(c.toArray());

            double perimeter = Imgproc.arcLength(c2, true);

            Mat hu = new Mat();
            Imgproc.HuMoments(moments, hu);

        }
    }

    public Mat adaptiveTresholding(Mat input, boolean treshMean) {
        Mat dst = new Mat(input.size(), input.type());
        Mat grey = new Mat(input.size(), input.type());

        Imgproc.cvtColor(input, grey, Imgproc.COLOR_BGR2GRAY);

        if (treshMean)
            Imgproc.adaptiveThreshold(grey, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 7, 2);
        else
            Imgproc.adaptiveThreshold(grey, dst, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 7, 2);

        Imgproc.cvtColor(dst, input, Imgproc.COLOR_GRAY2BGR);

        return input;
    }


}
