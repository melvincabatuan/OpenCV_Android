package vision.computer.opencv_android.contours;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.util.ArrayList;

import vision.computer.opencv_android.training.TrainingData;

/**
 * Created by Daniel on 04/04/2016.
 */
public class Contours {

    private static final int NEXT_IMAGE = 1;
    private static final int PREVIOUS_IMAGE = -1;
    private static final int CV_FILL_CONTOURS = -1;

    private static View view;
    private static String path;
    private static ArrayList<String> files;
    private static int nImageIndex;

    private TrainingData[] td = new TrainingData[5];

    public Contours(View view, String path) {
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
        this.path = Environment.getExternalStorageDirectory().getAbsolutePath() + path;
        File folder = new File(this.path);
        File[] listOfFiles = folder.listFiles();
        ArrayList<String> files = new ArrayList<String>();

        for (int i = 0; i < listOfFiles.length; i++) {
            if (listOfFiles[i].isFile() && !listOfFiles[i].getName().contains(".txt"))
                files.add(listOfFiles[i].getName());
        }
        return files;
    }

    public Mat gaussian(Mat src) {
        Imgproc.GaussianBlur(src, src, new Size(5, 5), 0, 0);
        return src;
    }

    public Mat sobel(Mat src, int type) {

        Mat rst = new Mat(src.size(), CvType.CV_8U);
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat grad_x, grad_y;

        grad_x = sobelHorizontal(src, type, false);
        grad_y = sobelVertical(src, type, false);
        Core.convertScaleAbs(grad_x, grad_x);
        Core.convertScaleAbs(grad_y, grad_y);
        Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, rst);

        Core.normalize(rst, rst, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        Imgproc.cvtColor(rst, rst, Imgproc.COLOR_GRAY2BGR);

        return rst;
    }

    public Mat sobelVertical(Mat src, int type, boolean show) {

        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        if (show) {
            src = gaussian(src);
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        }
        Mat grad_y = new Mat(src.size(), src.type());

        if (type == 0) {
            /// Gradient Y
            Imgproc.Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta);
        } else {
            /// Gradient Y
            Imgproc.Scharr(src, grad_y, ddepth, 0, 1, scale, delta, Core.BORDER_DEFAULT);
        }

        if (show) {
            Mat dst = new Mat(src.size(), CvType.CV_8U);
            for (int y = 0; y < dst.rows(); y++) {
                for (int x = 0; x < dst.cols(); x++) {
                    short a = (short) grad_y.get(y, x)[0];
                    dst.put(y, x, (a / 2) + 128);
                }
            }
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
            return dst;
        } else {
            return grad_y;
        }
    }

    public Mat sobelHorizontal(Mat src, int type, boolean show) {

        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;

        if (show) {
            src = gaussian(src);
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        }
        Mat grad_x = new Mat(src.size(), src.type());
        if (type == 0) {
            /// Gradient X
            Imgproc.Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta);
        } else {
            /// Gradient X
            Imgproc.Scharr(src, grad_x, ddepth, 1, 0, scale, delta, Core.BORDER_DEFAULT);
        }

        if (show) {
            Mat dst = new Mat(src.size(), CvType.CV_8U);
            for (int y = 0; y < dst.rows(); y++) {
                for (int x = 0; x < dst.cols(); x++) {
                    short a = (short) grad_x.get(y, x)[0];
                    dst.put(y, x, (a / 2) + 128);
                }
            }
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
            return dst;
        } else {
            return grad_x;
        }
    }


    public Mat sobelOrientation(Mat src, int type) {

        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = sobelHorizontal(src, type, false);
        Mat grad_y = sobelVertical(src, type, false);
        //Core.convertScaleAbs(grad_x, grad_x);
        //Core.convertScaleAbs(grad_y, grad_y);

        Mat orientation = new Mat(src.size(), CvType.CV_8U);

        for (int y = 0; y < grad_x.rows(); y++) {
            for (int x = 0; x < grad_y.cols(); x++) {
                short a = (short) grad_y.get(y, x)[0];
                short b = (short) grad_x.get(y, x)[0];
                float atan = (float) Core.fastAtan2(a, b);
                orientation.put(y, x, (atan / Math.PI) * 128);
            }
        }
        //Core.normalize(orientation,orientation,0,255,Core.NORM_MINMAX,CvType.CV_8U);
        Imgproc.cvtColor(orientation, orientation, Imgproc.COLOR_GRAY2BGR);

        return orientation;
    }

    public Mat sobelMagnitude(Mat src, int type) {

        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = sobelHorizontal(src, type, false);
        Mat grad_y = sobelVertical(src, type, false);

        Mat mag = new Mat(src.size(), CvType.CV_8U);
        for (int y = 0; y < grad_x.rows(); y++) {
            for (int x = 0; x < grad_y.cols(); x++) {
                short a = (short) grad_y.get(y, x)[0];
                short b = (short) grad_x.get(y, x)[0];
                mag.put(y, x, Math.sqrt(a * a + b * b));
            }
        }
        Imgproc.cvtColor(mag, mag, Imgproc.COLOR_GRAY2BGR);

        return mag;
    }


    public Mat scharr(Mat src) {
        return sobel(src, 1);
    }

    public Mat canny(Mat src) {

        /*
        ESTA MAL, NO VALE LA FUNCION Canny
         */
        Mat canny = new Mat();
        int min_threshold = 50;
        int ratio = 3;
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(src, canny, min_threshold, min_threshold * ratio);
        Imgproc.cvtColor(canny, canny, Imgproc.COLOR_GRAY2BGR);
        return canny;
    }

    public Mat Hough(Mat src, int threshold) {
        Mat lines = new Mat();
        int minLineSize = 30;
        int lineGap = 10;
        /*
            HoughLines(InputArray, OutputArray, double rho, double theta, int threshold, double srn=0, double stn=0 )
            Hough transformation
            A line in one picture is actually an edge. Hough transform scans the whole image and using a transformation
            that converts all white pixel cartesian coordinates in polar coordinates; the black pixels are left out.
            So you won't be able to get a line if you first don't detect edges, because HoughLines() don't know how
            to behave when there's a grayscale.
         */

        src = scharr(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Imgproc.HoughLinesP(src, lines, 1, Math.PI / 180, threshold,
                minLineSize, lineGap);

        for (int x = 0; x < lines.cols(); x++) {

            double[] vec = lines.get(0, x);

            double x1 = vec[0],
                    y1 = vec[1],
                    x2 = vec[2],
                    y2 = vec[3];

            //Starting point over the horizon and ending point below it or the other way around
            if ((y1 > (src.rows() / 2) && y2 < (src.rows() / 2)) ||
                    (y1 < (src.rows() / 2) && y2 < (src.rows() / 2))) {
                Point start = new Point(x1, y1);
                Point end = new Point(x2, y2);

                Imgproc.line(src, start, end, new Scalar(255, 0, 0), 3);
            }
        }

        return src;

        /*Mat grad_x = sobelHorizontal(src, 0, false);
        Mat grad_y = sobelVertical(src, 0, false);

        HashMap<Double[], Integer> votes = new HashMap<Double[], Integer>();

        //Horizonte
        int y0 = src.rows()/2;
        int x0 = 0;

        int y1 = src.rows()/2;
        int x1 = src.cols();

        for (int y = 0; y < src.rows(); y++) {
            for (int x = 0; x < src.cols(); x++) {
                float a = (float) grad_y.get(y, x)[0];
                float b = (float) grad_x.get(y, x)[0];
                float mag = (float) Math.sqrt(a * a + b * b);

                if (mag > threshold) {
                    float atan = Core.fastAtan2(a, b);
                    double theta = (float) ((atan / Math.PI) * 128);

                    double ro = a * Math.cos(theta) + b * Math.sin(theta);
                    Double[] key = new Double[]{theta, ro};

                    if (votes.containsKey(key)) {
                        int n = votes.get(key) + 1;
                        votes.remove(key);
                        votes.put(new Double[]{theta, ro}, n);
                    } else
                        votes.put(key, 1);
                }
            }
        }
        /*

        //BGR Mat
        src = canny(src);
        HashMap<Double[], Integer> votes = new HashMap<Double[], Integer>();

        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; i < src.cols(); j++) {
                if (Math.sqrt(i * i + j * j) >= threshold) {
                    float x = j - src.cols() / 2;
                    float y = src.rows() / 2 - i;
                    double theta = Core.fastAtan2(x, y);;
                    double ro = x * Math.cos(theta) + y * Math.sin(theta);

                }
            }
        }
*/
    }

}
