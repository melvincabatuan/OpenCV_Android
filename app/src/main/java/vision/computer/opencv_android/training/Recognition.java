package vision.computer.opencv_android.training;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.util.Log;
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
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Random;
import java.util.Scanner;

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
    private TrainingData[] td = new TrainingData[5];

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

    private static Descriptors getDescriptors(List<MatOfPoint> contours, String name,int n,boolean normal) {
        ArrayList<Descriptors> d = new ArrayList<Descriptors>();
        for (int i = 0; i < contours.size(); i++) {
            Descriptors dAux = new Descriptors();

            Moments moments = Imgproc.contourMoments(contours.get(i));

            double area = Imgproc.contourArea(contours.get(i));

            if (area < 150)
                continue;

            contours.get(i);
            MatOfPoint2f c2 = new MatOfPoint2f(contours.get(i).toArray());
            double perimeter = Imgproc.arcLength(c2, true);

            Mat hu = new Mat();
            //Imgproc.HuMoments(moments, hu);

            double[] HUmoments = new double[8];
            HUmoments=calculateHuMoments(moments);
            dAux.setArea(area);
            dAux.setPerimeter(perimeter);
            dAux.setHuMoments(HUmoments);
            dAux.setName(name);
            d.add(dAux);
        }
        if (d.size() > 1 && normal)
            return null;
        if(d.isEmpty())
            return null;
        return d.get(n);
    }

    public static double[] calculateHuMoments(Moments p){
        double[] moments= new double[8];
        double
                n20 = p.get_nu20(),
                n02 = p.get_nu02(),
                n30 = p.get_nu30(),
                n12 = p.get_nu12(),
                n21 = p.get_nu21(),
                n03 = p.get_nu03(),
                n11 = p.get_nu11();

        //First moment
        moments[0] = n20 + n02;

        //Second moment
        moments[1] = Math.pow((n20 - 02), 2) + Math.pow(2 * n11, 2);

        //Third moment
        moments[2] = Math.pow(n30 - (3 * (n12)), 2)
                + Math.pow((3 * n21 - n03), 2);

        //Fourth moment
        moments[3] = Math.pow((n30 + n12), 2) + Math.pow((n12 + n03), 2);

        //Fifth moment
        moments[4] = (n30 - 3 * n12) * (n30 + n12)
                * (Math.pow((n30 + n12), 2) - 3 * Math.pow((n21 + n03), 2))
                + (3 * n21 - n03) * (n21 + n03)
                * (3 * Math.pow((n30 + n12), 2) - Math.pow((n21 + n03), 2));

        //Sixth moment
        moments[5] = (n20 - n02)
                * (Math.pow((n30 + n12), 2) - Math.pow((n21 + n03), 2))
                + 4 * n11 * (n30 + n12) * (n21 + n03);

        //Seventh moment
        moments[6] = (3 * n21 - n03) * (n30 + n12)
                * (Math.pow((n30 + n12), 2) - 3 * Math.pow((n21 + n03), 2))
                + (n30 - 3 * n12) * (n21 + n03)
                * (3 * Math.pow((n30 + n12), 2) - Math.pow((n21 + n03), 2));

        //Eighth moment
        moments[7] = n11 * (Math.pow((n30 + n12), 2) - Math.pow((n03 + n21), 2))
                - (n20 - n02) * (n30 + n12) * (n03 + n21);

        return moments;
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

    public void training(String filename) throws IOException {
        File f = new File(this.path + filename);
        if (f.exists() && !f.isDirectory()) {
            TrainingData[] td = retrieveData(filename);
            for (int i = 0; i < td.length; i++) {
                if (td[i].getName().equals("circulo"))
                    this.td[0] = td[i];
                if (td[i].getName().equals("rectangulo"))
                    this.td[3] = td[i];
                if (td[i].getName().equals("vagon"))
                    this.td[1] = td[i];
                if (td[i].getName().equals("rueda"))
                    this.td[4] = td[i];
                if (td[i].getName().equals("triangulo"))
                    this.td[2] = td[i];
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
                        td[0].addDescriptors(getDescriptors(getContours(image), "circulo",0,true));
                    } else if (s.contains("triangulo")) {
                        td[2].addDescriptors(getDescriptors(getContours(image), "triangulo",0,true));
                    } else if (s.contains("rueda")) {
                        td[4].addDescriptors(getDescriptors(getContours(image), "rueda",0,true));
                    } else if (s.contains("vagon")) {
                        td[1].addDescriptors(getDescriptors(getContours(image), "vagon",0,true));
                    } else if (s.contains("rectangulo")) {
                        td[3].addDescriptors(getDescriptors(getContours(image), "rectangulo",0,true));
                    }
                }
            }

            for (int i = 0; i < td.length; i++)
                td[i].computeCalculations();

            storeData(new TrainingData[]{td[0], td[3], td[4], td[1], td[2]}, filename);
        }
    }

    public double[][] mahalanobisDistance(List<MatOfPoint> contours,int n) {
        Descriptors d = getDescriptors(contours, files.get(nImageIndex),n,false);
        double[][] result = new double[5][5];
        for (int i = 0; i< td.length; i++){
            result[i] = td[i].mahalanobisDistance(d);
        }

        return result;
    }

    public List<MatOfPoint> numberObjects(Mat input){
        return getBigContours(getContours(input));
    }

    public List<MatOfPoint> getBigContours(List<MatOfPoint> cont){
        Iterator<MatOfPoint> recorrer = cont.iterator();

        while (recorrer.hasNext()) {
            MatOfPoint c = recorrer.next();
            if (Imgproc.contourArea(c)<150){
                recorrer.remove();
            }
        }
        return cont;
    }

    private void storeData(TrainingData[] data, String fileName) throws IOException {
        OutputStreamWriter outputStreamWriter = new OutputStreamWriter(new FileOutputStream(this.path + fileName), "utf-8");
        for (TrainingData td : data) {
            outputStreamWriter.write(td.storeData());
            outputStreamWriter.write("\n");
        }
        outputStreamWriter.close();
    }

    private TrainingData[] retrieveData(String fileName) {
        int classes = 5;
        int descriptors = 5;
        int huMoments = 8;
        TrainingData[] td = new TrainingData[classes];
        try {
            Scanner scan = new Scanner(new FileInputStream(this.path + fileName),"utf-8");
            for (int u = 0; u < classes; u++) {
                String nam = scan.next();
                td[u] = new TrainingData(nam);
                for (int i = 0; i < descriptors; i++) {
                    Double mean = Double.parseDouble(scan.next());
                    td[u].setMean(mean, i);
                    Double var = Double.parseDouble(scan.next());
                    td[u].setVariance(var, i);
                    //Get descriptor i
                    Descriptors d = new Descriptors();
                    String n=scan.next();
                    Log.d("DBG","nam: "+n);
                    d.setName(n);
                    Double area = Double.parseDouble(scan.next());
                    d.setArea(area);
                    Double perim = Double.parseDouble(scan.next());
                    d.setPerimeter(perim);
                    double[] hu = new double[huMoments];

                    for (int j = 0; j < huMoments; j++) {
                        Double h = Double.parseDouble(scan.next());
                        hu[j] = h;
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
            if (listOfFiles[i].isFile() && !listOfFiles[i].getName().contains(".txt"))
                files.add(listOfFiles[i].getName());
        }
        return files;
    }
}
