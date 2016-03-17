package vision.computer.opencv_android;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

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

    /*
        0 = Circulo
        1 = Vagon
        2 = Triangulo
        3 = Rectangulo
        4 = Rueda
     */
    private static TrainingData[] td = new TrainingData[5];

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

    private static Descriptors getDescriptors(List<MatOfPoint> contours, String name) {
        ArrayList<Descriptors> d = new ArrayList<Descriptors>();
        for (int i = 0; i < contours.size(); i++) {
            Descriptors dAux = new Descriptors();

            Moments moments = Imgproc.moments(contours.get(i));

            double area = Imgproc.contourArea(contours.get(i));

            if (area < 200)
                break;

            //MatOfPoint2f c2 = new MatOfPoint2f(c.toArray());
            //double perimeter = Imgproc.arcLength(c2, true);

            Mat hu = new Mat();
            Imgproc.HuMoments(moments, hu);

            dAux.setArea(area);
            //dAux.setPerimeter(perimeter);
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

    private static List<MatOfPoint> getContours(Mat input) {
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        input = otsuThresholding(input, false);
        Imgproc.cvtColor(input, input, Imgproc.COLOR_BGR2GRAY);
        Imgproc.findContours(input, contours, new Mat(), Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);
        return contours;
    }

    public void training(String filename) {
        File f = new File(this.path + filename);
        if (f.exists() && !f.isDirectory()) {
            TrainingData[] td = retrieveData(filename);
            for (int i = 0; i < td.length; i++) {
                if (td[i].getName().equals("circulo"))
                    td[0] = td[i];
                if (td[i].getName().equals("rectangulo"))
                    td[3] = td[i];
                if (td[i].getName().equals("vagon"))
                    td[1] = td[i];
                if (td[i].getName().equals("rueda"))
                    td[4] = td[i];
                if (td[i].getName().equals("triangulo"))
                    td[2] = td[i];
            }
        } else {
            td[0] = new TrainingData("circulo");
            td[3] = new TrainingData("rectangulo");
            td[1] = new TrainingData("vagon");
            td[2] = new TrainingData("triangulo");
            td[4] = new TrainingData("rueda");

            for (String s : files) {
                if (!s.contains("reco")) {
                    Mat image = loadImage(s);
                    if (s.contains("circulo")) {
                        td[0].addDescriptors(getDescriptors(getContours(image), "circulo"));
                    } else if (s.contains("triangulo")) {
                        td[2].addDescriptors(getDescriptors(getContours(image), "triangulo"));
                    } else if (s.contains("rueda")) {
                        td[4].addDescriptors(getDescriptors(getContours(image), "rueda"));
                    } else if (s.contains("vagon")) {
                        td[1].addDescriptors(getDescriptors(getContours(image), "vagon"));
                    } else if (s.contains("rectangulo")) {
                        td[3].addDescriptors(getDescriptors(getContours(image), "rectangulo"));
                    }
                }
            }

            for (int i = 0; i < td.length; i++)
                td[i].computeCalculations();

            storeData(new TrainingData[]{td[0], td[3], td[4], td[1], td[3]}, filename);
        }
    }

    public double[][] mahalanobisDistance(Mat input) {
        List<MatOfPoint> contours = getContours(input);
        Descriptors d = getDescriptors(contours, files.get(nImageIndex));
        double[][] result = new double[5][5];
        for (int i = 0; i< td.length; i++)
            result[i] = td[i].mahalanobisDistance(d);
        return result;
    }

    private void storeData(TrainingData[] data, String fileName) {
        try {
            PrintWriter writer = new PrintWriter(new File(this.path + fileName), "UTF-8");
            for (TrainingData td : data) {
                writer.write(td.storeData() + "\n");
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (UnsupportedEncodingException e) {
            e.printStackTrace();
        }
    }

    private TrainingData[] retrieveData(String fileName) {
        int classes = 5;
        int descriptors = 5;
        int huMoments = 7;
        TrainingData[] td = new TrainingData[classes];
        try {
            Scanner scan = new Scanner(new File(this.path + fileName));
            for (int u = 0; u < classes; u++) {
                td[u].setName(scan.next());

                for (int i = 0; i < descriptors; i++) {
                    td[u].setMean(scan.nextDouble(), i);
                    td[u].setVariance(scan.nextDouble(), i);

                    //Get descriptor i
                    Descriptors d = new Descriptors();
                    d.setName(scan.next());
                    d.setArea(scan.nextDouble());
                    d.setPerimeter(scan.nextDouble());
                    double[] hu = new double[huMoments];

                    for (int j = 0; j < huMoments; j++) {
                        hu[j] = scan.nextDouble();
                    }

                    d.setHuMoments(hu);
                    td[u].addDescriptors(d);
                }
            }
            scan.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        return td;
    }

    private ArrayList<String> directories(String path) {
        this.path = Environment.getExternalStorageDirectory().getAbsolutePath() + path;
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
