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

/**
 * Created by Manuel on 09/03/2016.
 */
public class Recognition {

    private static View view;
    private static String path;
    private static ArrayList<String> files;
    private static int nImageIndex;

    public  Recognition(View view, String path) {
        this.view = view;
        files = directories(path);
        nImageIndex = -1;
    }

    public static Mat loadImage(boolean next) {
        if (android.os.Environment.getExternalStorageState().equals(
                android.os.Environment.MEDIA_MOUNTED)) {
            if (next) {
                if (nImageIndex == files.size() - 1)
                    nImageIndex = 0;
                else
                    nImageIndex++;
            } else {
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
        Imgproc.threshold(input, dst, 127, 255, Imgproc.THRESH_BINARY);
        return dst;
    }

    public Mat otsuThresholding(Mat input, boolean gaussian) {
        Mat dst = new Mat();
        if (gaussian)
            Imgproc.GaussianBlur(input, input, new Size(5, 5), 0);

        Imgproc.threshold(input, dst, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);

        return dst;
    }

    public Mat contours(Mat input) {
        Mat dst = new Mat();
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hierarchy = new Mat();
        Imgproc.Canny(input, dst, 50, 200);
        Imgproc.findContours(dst, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        for (int contourIdx = 0; contourIdx < contours.size(); contourIdx++) {
            Imgproc.drawContours(dst, contours, contourIdx, new Scalar(0, 0, 255), -1);
        }
        return dst;
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

    public Mat adaptiveTresholding(Mat input, boolean median, boolean treshMean) {
        Mat dst = new Mat();
        if (median)
            Imgproc.medianBlur(input, input, 5);

        if (treshMean)
            Imgproc.adaptiveThreshold(input, dst, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 11, 2);
        else
            Imgproc.adaptiveThreshold(input, dst, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 11, 2);

        return dst;
    }


}
