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
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

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
            Imgproc.cvtColor(src, src, Imgproc.COLOR_GRAY2BGR);
            return grad_y;
        }
    }

    public Mat sobelHorizontal(Mat src, int type, boolean show) {

        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        src = gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
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
            Imgproc.cvtColor(src, src, Imgproc.COLOR_GRAY2BGR);
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
                if (atan<0){
                    atan= (float) (2*Math.PI+atan);
                }
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

    private int cvRound(double x) {
        int y;
        if (x >= (int) x + 0.5)
            y = (int) x++;
        else
            y = (int) x;
        return y;
    }

    public Mat Hough(Mat src, int threshold) {
        Mat lines = new Mat();
        int minLineSize = 10;
        int lineGap = 10;

        Mat grads= src.clone();
        Mat grad_x = sobelHorizontal(grads, 0, false);
        Mat grad_y = sobelVertical(grads, 0, false);

        HashMap<Point, ArrayList<Line>> votes = new HashMap<Point, ArrayList<Line>>();

        //Horizonte
        float y0_h = src.rows()/2;
        float x0_h = 0;

        float y1_h = src.rows()/2;
        float x1_h = src.cols();

        for (int y = 0; y < src.rows(); y++) {
            for (int x = 0; x < src.cols(); x++) {
                float a1 = (float) grad_x.get(y, x)[0];
                float b1 = (float) grad_y.get(y, x)[0];
                float mag = (float) Math.sqrt(a1 * a1 + b1 * b1);

                if (mag > threshold) {
                    float atan = Core.fastAtan2(b1, a1);
                    double theta = (float) ((atan / Math.PI) * 128);

                    double ro = a1 * Math.cos(theta) + b1 * Math.sin(theta);

                    float a2=1;
                    float b2= (float) ((ro- a2*Math.cos(theta))/Math.sin(theta));

                    Point l1= new Point(a1,b1);
                    Point l2 = new Point(a2,b2);

                    Point intersection= intersection(x0_h,y0_h,x1_h,y1_h,a1,b1,a2,b2,src.rows(),src.cols());

                    //p=x*cos(theta) + y * sen(theta)
                    if (intersection!=null) {
                        if (votes.containsKey(intersection)) {
                            ArrayList<Line> lineArray = votes.get(intersection);
                            lineArray.add(new Line(l1,l2));
                            votes.remove(intersection);
                            votes.put(intersection, lineArray);
                        } else {
                            ArrayList<Line> lineArray = new ArrayList<>();
                            lineArray.add(new Line(l1, l2));
                            votes.put(intersection, lineArray);
                        }
                    }
                }
            }
        }

        Point fuga = null;
        ArrayList<Line> lineasFuga=null;
        int maxValue=0;
        Iterator it = votes.entrySet().iterator();
        while (it.hasNext()) {
            Map.Entry<Point,ArrayList<Line>> pair = (Map.Entry)it.next();
            if(pair.getValue().size()>maxValue){
                fuga=pair.getKey();
                lineasFuga=pair.getValue();
                maxValue=pair.getValue().size();
                System.out.println(pair.getKey() + " = " + pair.getValue().size());
            }
            it.remove();

        }
        for(Line l : lineasFuga){
            Log.d("DBG", "lineas: "+l.getPoint1()+"  "+l.getPoint2());
            Imgproc.line(src,l.getPoint1(),l.getPoint2(), new Scalar (255,0,0),1);
        }
        Imgproc.line(src,new Point(x0_h,y0_h),new Point(x1_h,y1_h), new Scalar (0,0,255),2);
        Imgproc.circle(src,fuga,3,new Scalar(255, 0, 0),3);
        return src;
    }

    public Point intersection(float x1, float y1, float x2, float y2, float x3, float y3, float x4, float y4,int fils, int cols) {
        float d = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
        if (d == 0) return null;

        float xi = ((x3-x4)*(x1*y2-y1*x2)-(x1-x2)*(x3*y4-y3*x4))/d;
        float yi = ((y3-y4)*(x1*y2-y1*x2)-(y1-y2)*(x3*y4-y3*x4))/d;

        if(xi<0 || xi>cols || yi<0 || yi>fils)
            return null;

        return new Point(xi,yi);
    }

}
