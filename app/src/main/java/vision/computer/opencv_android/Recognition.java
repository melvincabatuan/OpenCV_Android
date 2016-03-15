package vision.computer.opencv_android;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.CvType;
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

import vision.computer.opencv_android.training.Descriptors;
import vision.computer.opencv_android.training.TrainingData;

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

    private static TrainingData tdCirculo = new TrainingData();
    private static TrainingData tdRectangulo = new TrainingData();
    private static TrainingData tdVagon = new TrainingData();
    private static TrainingData tdTriangulo = new TrainingData();
    private static TrainingData tdRueda = new TrainingData();

    public Recognition(View view, String path) {
        this.view = view;
        files = directories(path);
        nImageIndex = 0;
    }

    public static Mat loadImage(String fileName) {
        if (android.os.Environment.getExternalStorageState().equals(
                android.os.Environment.MEDIA_MOUNTED)) {
            Mat image = Imgcodecs.imread(path + fileName);
            return image;
        }
        return null;
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

    public static void training() {
        Mat training = new Mat(5, 5, CvType.CV_32FC1);
        ArrayList<Descriptors> trainingData = new ArrayList<Descriptors>();

        for (String s : files) {
            if (!s.contains("reco")) {
                Mat image = loadImage(s);
                Imgproc.cvtColor(image,image,Imgproc.COLOR_BGR2GRAY);
                if (s.contains("circulo")) {
                    tdCirculo.addDescriptors(getDescriptors(getContours(image), "circulo"));
                } else if (s.contains("triangulo")) {
                    tdTriangulo.addDescriptors(getDescriptors(getContours(image), "triangulo"));
                } else if (s.contains("rueda")) {
                    tdRueda.addDescriptors(getDescriptors(getContours(image), "rueda"));
                } else if (s.contains("vagon")) {
                    tdVagon.addDescriptors(getDescriptors(getContours(image), "vagon"));
                } else if (s.contains("rectangulo")) {
                    tdRectangulo.addDescriptors(getDescriptors(getContours(image), "rectangulo"));
                }
            }
        }

        tdCirculo.computeCalculations();
        tdTriangulo.computeCalculations();
        tdRueda.computeCalculations();
        tdVagon.computeCalculations();
        tdRectangulo.computeCalculations();
    }

    private static Descriptors getDescriptors(List<MatOfPoint> contours, String name) {
        ArrayList<Descriptors> d = new ArrayList<Descriptors>();
        for (int i = 0; i < contours.size(); i++) {
            Descriptors dAux = new Descriptors();
            MatOfPoint c = contours.get(i);
            Moments moments = Imgproc.moments(c);
            double area = Imgproc.contourArea(c);

            if (area < 200)
                break;

            MatOfPoint2f c2 = new MatOfPoint2f(c.toArray());
            double perimeter = Imgproc.arcLength(c2, true);

            Mat hu = new Mat();
            Imgproc.HuMoments(moments, hu);

            dAux.setArea(area);
            dAux.setPerimeter(perimeter);
            dAux.setHuMoments(hu.get(0, 0));
            dAux.setName(name);
        }
        if (d.size() > 1)
            return null;
        return d.get(0);
    }

    public static Mat regularTresholding(Mat input) {
        Mat dst = new Mat();
        Imgproc.threshold(input, dst, 127, 255, Imgproc.THRESH_BINARY_INV);
        return dst;
    }

    public static Mat otsuThresholding(Mat input, boolean gaussian) {
        Mat dst = new Mat(input.size(), input.type());
        Mat gray = new Mat(input.size(), input.type());

        if (gaussian)
            Imgproc.GaussianBlur(input, input, new Size(5, 5), 0);

        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(gray, dst, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        Imgproc.cvtColor(dst, input, Imgproc.COLOR_GRAY2BGR);
        return input;
    }

    private static List<MatOfPoint> getContours(Mat input) {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        input = otsuThresholding(input, false);
        Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2GRAY);
        Imgproc.findContours(input, contours, null, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }

    public static Mat contours(Mat input) {
        Random r = new Random();
        Mat gray = new Mat(input.size(), input.type());
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Mat hier = new Mat();

        //Treshold the image --> less errors/noise
        input = otsuThresholding(input, false);
        Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY);

        //Find contours and change color range to BGR
        Imgproc.findContours(gray, contours, hier, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        Imgproc.cvtColor(gray, input, Imgproc.COLOR_GRAY2BGR);

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

    public static Mat adaptiveTresholding(Mat input, boolean treshMean) {
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
}
