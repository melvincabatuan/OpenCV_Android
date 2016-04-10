package vision.computer.opencv_android.contours;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;
import java.util.Random;

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

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
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
            Imgproc.Scharr(src, grad_y, ddepth, 0, 1, scale, delta);
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
            Imgproc.Scharr(src, grad_x, ddepth, 1, 0, scale, delta);
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

    private Point intersection(Point o1, Point p1, Point o2, Point p2)

    {
        Point x = new Point(o2.x - o1.x, o2.y - o1.y);
        Point d1 = new Point(p1.x - o1.x, p1.y - o1.y);
        Point d2 = new Point(p2.x - o2.x, p2.y - o2.y);

        double cross = d1.x * d2.y - d1.y * d2.x;
        if (Math.abs(cross) < /*EPS*/1e-8)
            return null;

        double t1 = (x.x * d2.y - x.y * d2.x) / cross;
        Point r = new Point(o1.x + d1.x * t1, o1.y + d1.y + t1);
        return r;
    }

    public Mat Hough(Mat src, int threshold) {

        Mat grad_x = sobelHorizontal(src, 0, false);
        Mat grad_y = sobelVertical(src, 0, false);

        HashMap<Double[], Integer> votes = new HashMap<Double[], Integer>();
        Rect imageRect = new Rect(new Point(0, 0), new Point(src.rows(), src.cols()));

        //Horizonte
        int y0 = src.rows() / 2;
        int x0 = 0;
        int x1 = src.cols();

        Point s0 = new Point(x0, y0);
        Point e0 = new Point(x1, y0);

        Imgproc.clipLine(imageRect, s0, e0);
        Imgproc.line(src, s0, e0, new Scalar(255, 0, 0), 1);

        for (int y = 0; y < src.rows(); y++) {
            for (int x = 0; x < src.cols(); x++) {
                float a = (float) grad_y.get(y, x)[0];
                float b = (float) grad_x.get(y, x)[0];
                float mag = (float) Math.sqrt(a * a + b * b);

                if (mag > threshold) {

                    float atan = Core.fastAtan2(a, b);

                    //Se hace el calculo de Theta
                    double theta = (float) ((atan / Math.PI) * 128);

                    double cos = Math.cos(theta);
                    double sin = Math.sin(theta);

                    //Se hace el calculo de Rho
                    double rho = a * cos + b * sin;

                    double iniX = a * rho;
                    double iniY = b * rho;

                    /*
                    Se obtienen dos puntos de la recta que corta por a y b usando la ecuacion de la recta
                     */
                    Point p1 = new Point(iniX + 1000 * -sin, iniY + 1000 * cos);
                    Point p2 = new Point(iniX - 1000 * -sin, iniY - 1000 * cos);

                    /*
                    Se comprueba que efectivamente ese punto intersecta con la linea del horizonte
                     */
                    Point inters = intersection(p1, p2, s0, e0);
                    if (inters != null && inters.x <= src.cols() && inters.y >= src.rows()
                            && inters.x >= 0 && inters.y >= 0) {
                        Double[] key = new Double[]{round(theta, 2), round(rho, 2)};

                        if (votes.containsKey(key)) {
                            int n = votes.get(key) + 1;
                            votes.remove(key);
                            votes.put(new Double[]{theta, rho}, n);
                        } else
                            votes.put(key, 1);
                    }
                }
            }
        }
        Iterator it = votes.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Double[], Integer> pair = (Map.Entry) it.next();
            //Se obtienen las rectas que obtienen un nº de votos mayor que X
            if (pair.getValue() > 0) {
                //Se hace el calculo de Theta
                double theta = pair.getKey()[0];

                double cos = Math.cos(theta);
                double sin = Math.sin(theta);

                //Se hace el calculo de Rho
                double rho = pair.getKey()[1];
                double a = new Random().nextDouble()*10;
                double b = (rho - a*cos)/sin;
                double iniX = a * rho;
                double iniY = b * rho;

                /*
                Se obtienen dos puntos de la recta que corta por a y b usando la ecuacion de la recta
                 */
                Point p1 = new Point(iniX + 1000 * -sin, iniY + 1000 * cos);
                Point p2 = new Point(iniX - 1000 * -sin, iniY - 1000 * cos);

                /*
                Se dibuja esa recta
                 */
                Imgproc.clipLine(imageRect, p1, p2);
                Imgproc.line(src, p1, p2, new Scalar(255, 0, 0), 1);
            }
            it.remove(); // avoids a ConcurrentModificationException
        }
        return src;
    }
      /*Mat lines = new Mat();
        //Uso en Probabilistic Hough Transform
        int minLineSize = 30;
        int lineGap = 10;

            HoughLines(InputArray, OutputArray, double rho, double theta, int threshold, double srn=0, double stn=0 )
            Hough transformation
            A line in one picture is actually an edge. Hough transform scans the whole image and using a transformation
            that converts all white pixel cartesian coordinates in polar coordinates; the black pixels are left out.
            So you won't be able to get a line if you first don't detect edges, because HoughLines() don't know how
            to behave when there's a grayscale.


        src = scharr(src);
        Rect imageRect = new Rect(new Point(0,0), new Point(src.rows(),src.cols()));
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Imgproc.HoughLines(src, lines, 1, Math.PI / 90, threshold);

        for (int x = 0; x < lines.cols(); x++) {

            /*double rho = lines.get(x, 0)[0];
            double theta = lines.get(x, 1)[0];

            double a = Math.cos(theta);
            double b = Math.sin(theta);
            double x0 = a * rho;
            double y0 = b * rho;

            Point p1 = new Point(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
            Point p2 = new Point(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));

            double[] vec = lines.get(0, x);
            double x1 = vec[0],
                    y1 = vec[1],
                    x2 = vec[2],
                    y2 = vec[3];
            Point p1 = new Point(x1, y1);
            Point p2 = new Point(x2, y2);

            //Starting point over the horizon and ending point below it or the other way around
            if ((p1.y > (src.rows() / 2) && p2.y < (src.rows() / 2)) ||
                    (p1.y < (src.rows() / 2) && p2.y < (src.rows() / 2))) {
                Imgproc.clipLine(imageRect,p1,p2);
                Imgproc.line(src, p1, p2, new Scalar(255, 0, 0), 1);
            }
        }

        return src;
*/
}
