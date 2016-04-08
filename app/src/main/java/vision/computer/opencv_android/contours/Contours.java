package vision.computer.opencv_android.contours;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
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

    public Mat gaussian(Mat src){
        Imgproc.GaussianBlur(src, src, new Size(3,3), 0, 0, Core.BORDER_DEFAULT);
        return src;
    }
    public Mat sobel(Mat src, int type) {

        Mat rst = new Mat();
        src=gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        Mat grad_x, grad_y;
        Mat abs_grad_x = new Mat(), abs_grad_y = new Mat();

        grad_x=sobelHorizontal(src,type,false);
        grad_y=sobelVertical(src,type,false);

        Core.convertScaleAbs(grad_x, abs_grad_x);
        Core.convertScaleAbs(grad_y, abs_grad_y);
        Core.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, rst);

        Imgproc.cvtColor(rst, rst, Imgproc.COLOR_GRAY2BGR);

        return rst;
    }

    public Mat sobelVertical(Mat src, int type,boolean show) {

        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_16S;
        if(show){
            src=gaussian(src);
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        }
        Mat grad_y = new Mat();

        if (type==0){
            /// Gradient Y
            Imgproc.Sobel(src, grad_y, ddepth, 0, 1, 3, scale, delta, Core.BORDER_DEFAULT);
        }
        else{
            /// Gradient Y
            Imgproc.Scharr(src, grad_y, ddepth, 0, 1, scale, delta, Core.BORDER_DEFAULT);
        }

        if(show){
            /*Core.multiply(grad_y,new Scalar(0.5),grad_y);
            Core.add(grad_y,new Scalar(128),grad_y);*/
            Core.MinMaxLocResult res = Core.minMaxLoc(grad_y);
            grad_y.convertTo(grad_y, CvType.CV_8U, 255.0 / (res.maxVal - res.minVal), -res.minVal * 255.0 / (res.maxVal - res.minVal));
            Core.bitwise_not(grad_y, grad_y);
            Imgproc.cvtColor(grad_y, grad_y, Imgproc.COLOR_GRAY2BGR);
            return grad_y;
        }else{
            return grad_y;
        }
    }

    public Mat sobelHorizontal(Mat src, int type, boolean show) {

        int scale = 1;
        int delta = 0;
        int ddepth = CvType.CV_32F;
        Mat grad_x = new Mat();

        if(show){
            src=gaussian(src);
            Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
        }

        if (type==0){
            /// Gradient X
            Imgproc.Sobel(src, grad_x, ddepth, 1, 0, 3, scale, delta, Core.BORDER_DEFAULT);
        }
        else{
            /// Gradient X
            Imgproc.Scharr(src, grad_x, ddepth, 1, 0, scale, delta, Core.BORDER_DEFAULT);
        }

        if(show){
            //Core.multiply(grad_x,new Scalar(0.5),grad_x);
            //Core.add(grad_x,new Scalar(128),grad_x);
            //Core.normalize(grad_x,grad_x,0,255,Core.NORM_MINMAX,CvType.CV_8U);
            Core.MinMaxLocResult res = Core.minMaxLoc(grad_x);
            grad_x.convertTo(grad_x, CvType.CV_8U, 255.0 / (res.maxVal - res.minVal), -res.minVal * 255.0 / (res.maxVal - res.minVal));
            Core.bitwise_not(grad_x, grad_x);
            Imgproc.cvtColor(grad_x, grad_x, Imgproc.COLOR_GRAY2BGR);
            return grad_x;
        }else{
            return grad_x;
        }
    }


    public Mat sobelOrientation(Mat src, int type) {

        //src=gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat abs_grad_x=new Mat(),abs_grad_y=new Mat();
        Mat grad_x = sobelHorizontal(src,type,false);
        Mat grad_y = sobelVertical(src,type,false);

        Mat orientation= new Mat(grad_x.rows(),grad_y.cols(),CvType.CV_32F);
        grad_x.convertTo(grad_x,CvType.CV_32F);
        grad_y.convertTo(grad_y,CvType.CV_32F);

        for(int y=0;y<grad_x.rows();y++)
        {
            for(int x=0;x<grad_y.cols();x++)
            {
                float a=(float)grad_y.get(y,x)[0];
                float b=(float)grad_x.get(y,x)[0];
                float atan= Core.fastAtan2(a,b);
                orientation.put(y,x,atan);
            }
        }

        Core.normalize(orientation,orientation,0,255,Core.NORM_MINMAX,CvType.CV_8U);
                //cv::normalize(orientation, orientation, 0x00, 0xFF, cv::NORM_MINMAX, CV_8U);
        Imgproc.cvtColor(orientation, orientation, Imgproc.COLOR_GRAY2BGR);

        return orientation;
    }

    public Mat sobelMagnitude(Mat src, int type) {

        //src=gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = sobelHorizontal(src,type,false);
        Mat grad_y = sobelVertical(src,type,false);
        grad_x.convertTo(grad_x,CvType.CV_32F);
        grad_y.convertTo(grad_y,CvType.CV_32F);

        Mat mag= new Mat(grad_x.rows(),grad_y.cols(),CvType.CV_32F);
        Core.magnitude(grad_x,grad_y,mag);

        Core.normalize(mag,mag,0,255,Core.NORM_MINMAX,CvType.CV_8U);

        Imgproc.cvtColor(mag, mag, Imgproc.COLOR_GRAY2BGR);

        return mag;
    }


    public Mat scharr(Mat src) {
        return sobel(src,1);
    }

    public Mat canny (Mat src) {
        Mat canny = new Mat();
        int min_threshold=50;
        int ratio = 3;
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(src, canny, min_threshold, min_threshold * ratio);
        Imgproc.cvtColor(canny, canny, Imgproc.COLOR_GRAY2BGR);
        return canny;
    }
}
