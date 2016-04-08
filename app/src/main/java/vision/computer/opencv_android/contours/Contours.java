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
        int ddepth = CvType.CV_32F;
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
            Mat dst= new Mat(grad_y.size(),grad_y.type());
            for(int y=0;y<grad_y.rows();y++)
            {
                for(int x=0;x<grad_y.cols();x++)
                {
                    float a=(float)grad_y.get(y,x)[0];
                    dst.put(y,x,(a/2)+128);
                }
            }
            //Core.MinMaxLocResult res = Core.minMaxLoc(grad_y);
            //grad_y.convertTo(grad_y, CvType.CV_8U, 255.0 / (res.maxVal - res.minVal), -res.minVal * 255.0 / (res.maxVal - res.minVal));
            //Core.bitwise_not(grad_y, grad_y);
            Core.normalize(dst,dst,0,255,Core.NORM_MINMAX,CvType.CV_8U);
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
            return dst;
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
        Core.convertScaleAbs(grad_x, grad_x);1

        if(show){
            Mat dst= new Mat(grad_x.size(),grad_x.type());
            for(int y=0;y<grad_x.rows();y++)
            {
                for(int x=0;x<grad_x.cols();x++)
                {
                    float a=(float)grad_x.get(y,x)[0];
                    dst.put(y,x,(a/2)+128);
                }
            }
            Core.bitwise_not(dst, dst);
            Core.normalize(dst,dst,0,255,Core.NORM_MINMAX,CvType.CV_8U);
            Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
            return dst;
        }else{
            return grad_x;
        }
    }


    public Mat sobelOrientation(Mat src, int type) {

        src=gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat abs_grad_x=new Mat(),abs_grad_y=new Mat();
        Mat grad_x = sobelHorizontal(src,type,false);
        Mat grad_y = sobelVertical(src,type,false);
        Core.convertScaleAbs(grad_x, grad_x);
        Core.convertScaleAbs(grad_y, grad_y);

        Mat orientation= new Mat(grad_x.rows(),grad_y.cols(),CvType.CV_32F);

        for(int y=0;y<grad_x.rows();y++)
        {
            for(int x=0;x<grad_y.cols();x++)
            {
                float a=(float)grad_y.get(y,x)[0];
                float b=(float)grad_x.get(y,x)[0];
                float atan= Core.fastAtan2(a,b);
                orientation.put(y,x,(atan/Math.PI)*128);
            }
        }

        Core.normalize(orientation,orientation,0,255,Core.NORM_MINMAX,CvType.CV_8U);
        Imgproc.cvtColor(orientation, orientation, Imgproc.COLOR_GRAY2BGR);

        return orientation;
    }

    public Mat sobelMagnitude(Mat src, int type) {

        src=gaussian(src);
        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);

        Mat grad_x = sobelHorizontal(src,type,false);
        Mat grad_y = sobelVertical(src,type,false);
        Core.convertScaleAbs(grad_x, grad_x);
        Core.convertScaleAbs(grad_y, grad_y);

        Mat mag= new Mat(grad_x.size(),CvType.CV_32F);
        for(int y=0;y<grad_x.rows();y++)
        {
            for(int x=0;x<grad_y.cols();x++)
            {
                float a=(float)grad_y.get(y,x)[0];
                float b=(float)grad_x.get(y,x)[0];
                mag.put(y,x,Math.sqrt(a*a+b*b));
            }
        }

        Core.normalize(mag,mag,0,255,Core.NORM_MINMAX,CvType.CV_8U);

        Imgproc.cvtColor(mag, mag, Imgproc.COLOR_GRAY2BGR);

        return mag;
    }


    public Mat scharr(Mat src) {
        return sobel(src,1);
    }

    public Mat canny (Mat src) {

        /*
        ESTA MAL, NO VALE LA FUNCION Canny
         */
        Mat canny = new Mat();
        int min_threshold=50;
        int ratio = 3;
        Imgproc.cvtColor(src, src, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(src, canny, min_threshold, min_threshold * ratio);
        Imgproc.cvtColor(canny, canny, Imgproc.COLOR_GRAY2BGR);
        return canny;
    }

    public Mat Hough(Mat src) {

        /*
        1- Trazar una línea en el horizonte, que será un vector horizontal
        2- Recorrer todos los pixeles de la imagen y comprobar la magnitud de su gradiente,
            si supera un cierto threshold la magnitud (por ejemplo 25), entonces miramos su dirección
            y trazamos una línea.
        3- Si intersecta con el horizonte, aumentamos el contador de la posicion del vector donde haya intersectado
        4- Al final--> la posición del vector con más votos/intersecciones es el punto de fuga.
         */

        return src;
    }

}
