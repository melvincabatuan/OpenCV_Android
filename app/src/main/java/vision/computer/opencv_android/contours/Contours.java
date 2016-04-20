package vision.computer.opencv_android.contours;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.util.Log;
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
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

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

    private static int DDEPTH_SOBEL = CvType.CV_16S;

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
        Imgproc.GaussianBlur(src, src, new Size(3, 3), 0, 0, Core.BORDER_DEFAULT);
        return src;
    }

    public Mat sobel(Mat src, int type) {

        Mat rst = new Mat(src.size(), CvType.CV_8U);
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat grad_x = new Mat(), grad_y = new Mat();

        Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 0, 1, 3, 1, 0);
        Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, -1, 0);

        Core.convertScaleAbs(grad_x, grad_x);
        Core.convertScaleAbs(grad_y, grad_y);
        Core.addWeighted(grad_x, 0.5, grad_y, 0.5, 0, rst);

        Core.normalize(rst, rst, 0, 255, Core.NORM_MINMAX, CvType.CV_8U);
        Imgproc.cvtColor(rst, rst, Imgproc.COLOR_GRAY2BGR);

        return rst;
    }

    public Mat sobelVertical(Mat src, int type) {

        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_y = new Mat(src.size(), src.type());

        if (type == 0) {
            /// Gradient Y
            Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, -1, 0);
        } else {
            /// Gradient Y
            Imgproc.Scharr(src, grad_y, DDEPTH_SOBEL, 0, 1, -1, 0);
        }

        Mat dst = new Mat(src.size(), CvType.CV_8U);
        for (int y = 0; y < dst.rows(); y++) {
            for (int x = 0; x < dst.cols(); x++) {
                short a = (short) grad_y.get(y, x)[0];
                dst.put(y, x, (a / 2) + 128);
            }
        }
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
        return dst;

    }

    public Mat sobelHorizontal(Mat src, int type) {

        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat grad_x = new Mat(src.size(), src.type());
        if (type == 0) {
            /// Gradient X
            Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 1, 0, 3, 1, 0);
        } else {
            /// Gradient X
            Imgproc.Scharr(src, grad_x, DDEPTH_SOBEL, 1, 0, 1, 0);
        }

        Mat dst = new Mat(src.size(), CvType.CV_8U);
        for (int y = 0; y < dst.rows(); y++) {
            for (int x = 0; x < dst.cols(); x++) {
                short a = (short) grad_x.get(y, x)[0];
                dst.put(y, x, (a / 2) + 128);
            }
        }
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
        return dst;
    }

    public Mat sobelOrientation(Mat src, int type) {

        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = new Mat();
        Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 1, 0, 3, 1, 0);
        Mat grad_y = new Mat();
        Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, -1, 0);

        Mat orientation = new Mat(src.size(), CvType.CV_8U);

        for (int y = 0; y < grad_x.rows(); y++) {
            for (int x = 0; x < grad_y.cols(); x++) {
                short a = (short) grad_y.get(y, x)[0];
                short b = (short) grad_x.get(y, x)[0];
                float atandeg = Core.fastAtan2(a, b);
                float atan = (float) (atandeg * (Math.PI / 180));

                orientation.put(y, x, (atan / Math.PI) * 128);
            }
        }
        Imgproc.cvtColor(orientation, orientation, Imgproc.COLOR_GRAY2BGR);

        return orientation;
    }

    public Mat sobelMagnitude(Mat src, int type) {

        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = new Mat();
        Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 1, 0, 3, 1, 0);
        Mat grad_y = new Mat();
        Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, -1, 0);

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

    private int cvRound(double x) {
        int y;
        if (x >= (int) x + 0.5)
            y = (int) x++;
        else
            y = (int) x;
        return y;
    }

    public Mat Hough(Mat src, int threshold) {
        double var = 0.1;
        double yline1 = -1000;
        double yline2 = 1000;

        Mat draw = src.clone();
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = new Mat();
        Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 1, 0, 3, 1, 0);
        Mat grad_y = new Mat();
        Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, 1, 0);

        HashMap<Integer, ArrayList<Line>> votes = new HashMap<Integer, ArrayList<Line>>();

        int[] votationVector = new int[src.cols()];

        //Horizonte
        int y_h = src.rows() / 2;
        int x_h = 0;

        for (int y = 0; y < src.rows(); y++) {
            for (int x = 0; x < src.cols(); x++) {
                float a1 = (float) grad_x.get(y, x)[0];
                float b1 = (float) grad_y.get(y, x)[0];
                float mag = (float) Math.sqrt(a1 * a1 + b1 * b1);

                if (mag >= threshold) {
                    double atandeg = Core.fastAtan2(b1, a1);
                    double theta = (atandeg * (Math.PI / 180));     /*Pasar grados a radianes*/

                    double sinO = Math.sin(theta);
                    double cosO = Math.cos(theta);

                    if (Math.abs(sinO) > var && Math.abs(cosO) > var) {
                        double ro = x * cosO + y * sinO;
                        int xp = cvRound((ro - src.rows() / 2 * sinO) / cosO);
                        if (xp >= 0 && xp < src.cols()) {

                            /**
                             * El punto de contorno vota su interseccion con horizonte
                             */
                            votationVector[xp] = votationVector[xp] + 1;

                            //double xLine1=( ro - yline1 * sinO ) / cosO;
                            //double xLine2=( ro - yline2 * sinO ) / cosO;

/*                            if (votes.containsKey(xp)) {
                                ArrayList<Line> lineArray = votes.get(xp);
                                lineArray.add(new Line(new Point (xLine1,yline1),new Point (xLine2,yline2)));
                                votes.remove(xp);
                                votes.put(xp, lineArray);
                            } else {
                                ArrayList<Line> lineArray = new ArrayList<>();
                                lineArray.add(new Line(new Point (xLine1,yline1),new Point (xLine2,yline2)));
                                votes.put(xp, lineArray);
                            }*/
                        }
                    }
                }
            }
        }

        for (int i = 1; i < src.cols(); i++) {
            if (votationVector[x_h] < votationVector[i]) {
                x_h = i;
                Log.d("DBG", "vv: " + votationVector[x_h]);
            }
        }

        //ArrayList<Line> lineasFuga=votes.get(x_h);
        /*for(Line l : lineasFuga){
            Log.d("DBG", "lineas: "+l.getPoint1()+"  "+l.getPoint2());
           Imgproc.line(draw,l.getPoint1(),l.getPoint2(), new Scalar (255,0,0),2);
        }*/

        Log.d("DBG", "vv: " + votationVector[x_h]);
        Log.d("RES", "Punto de fuga: " + x_h + "," + y_h);

        Imgproc.line(draw, new Point(0, src.rows() / 2), new Point(src.cols(), src.rows() / 2), new Scalar(0, 0, 255), 2);
        Imgproc.line(draw, new Point(x_h - 10, y_h + 10), new Point(x_h + 10, y_h - 10), new Scalar(0, 0, 255), 1);
        Imgproc.line(draw, new Point(x_h - 10, y_h - 10), new Point(x_h + 10, y_h + 10), new Scalar(0, 0, 255), 1);

        return draw;
    }

    public Mat HoughLive(Mat src, int threshold) {
        double var = 0.3;
        double yline1 = -1000;
        double yline2 = 1000;

        Mat draw = src.clone();
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = new Mat();
        Imgproc.Sobel(src, grad_x, DDEPTH_SOBEL, 1, 0, 3, 1, 0);
        Mat grad_y = new Mat();
        Imgproc.Sobel(src, grad_y, DDEPTH_SOBEL, 0, 1, 3, 1, 0);

        HashMap<Integer, ArrayList<Line>> votations = new HashMap<Integer, ArrayList<Line>>();

        int[] votationVector = new int[src.cols()];
        int[][] votes = new int[src.rows()][src.cols()];

        //Horizonte
        int y_h = src.rows() / 2;
        int x_h = 0;

        for (int y = 0; y < src.rows(); y++) {
            for (int x = 0; x < src.cols(); x++) {
                float a1 = (float) grad_x.get(y, x)[0];
                float b1 = (float) grad_y.get(y, x)[0];
                float mag = (float) Math.sqrt(a1 * a1 + b1 * b1);

                if (mag >= threshold) {
                    double atandeg = Core.fastAtan2(b1, a1);
                    double theta = (atandeg * (Math.PI / 180));     /*Pasar grados a radianes*/
                    //theta = (theta / Math.PI) * 128;

                    double sinO = Math.sin(theta);
                    double cosO = Math.cos(theta);

                    if (Math.abs(sinO) > var && Math.abs(cosO) > var) {
                        double ro = x * cosO + y * sinO;
                        for (int yb = 0; yb < src.rows(); yb++) {
                            int y_int = yb;
                            int x_int = cvRound((ro - y_int * sinO) / cosO);

                            if (x_int >= 0 && x_int < src.cols()) {
                                /* Realiza la votacion al punto de interseccion */
                                votes[y_int][x_int] = votes[y_int][x_int] + 1;

                            }
                        }
                    }
                }
            }
        }
        int max_i = 0;
        int max_j = 0;
        for (int i = 1; i < src.rows(); i++) {
            for (int j = 1; j < src.cols(); j++) {
                if (votes[max_i][max_j] < votes[i][j]) {
                    max_i = i;
                    max_j = j;
                    Imgproc.line(draw, new Point(max_j - 1, max_i + 1), new Point(max_j + 1, max_i - 1), new Scalar(0, 255,0), 1);
                    Imgproc.line(draw, new Point(max_j - 1, max_i - 1), new Point(max_j + 1, max_i + 1), new Scalar(0, 255,0), 1);
                }
            }
        }
        Log.d("RES", "Punto de fuga: " + max_j + "," + max_i);

        Imgproc.line(draw, new Point(max_j - 10, max_i + 10), new Point(max_j + 10, max_i - 10), new Scalar(0, 0, 255), 1);
        Imgproc.line(draw, new Point(max_j - 10, max_i - 10), new Point(max_j + 10, max_i + 10), new Scalar(0, 0, 255), 1);

        return draw;
    }

    public Point intersection(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4, int fils, int cols) {
        float d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
        if (d == 0) return null;

        float xi = ((x3 - x4) * (x1 * y2 - y1 * x2) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d;
        float yi = ((y3 - y4) * (x1 * y2 - y1 * x2) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d;

        if (xi < 0 || xi > cols || yi < 0 || yi > fils)
            return null;

        return new Point(xi, yi);
    }

    public static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();

        long factor = (long) Math.pow(10, places);
        value = value * factor;
        long tmp = Math.round(value);
        return (double) tmp / factor;
    }

}
