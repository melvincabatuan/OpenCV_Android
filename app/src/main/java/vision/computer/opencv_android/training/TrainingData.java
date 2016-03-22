package vision.computer.opencv_android.training;

import org.opencv.core.Mat;

import java.util.ArrayList;

/**
 * Created by Manuel on 14/03/2016.
 */
public class TrainingData {

    private String name;
    private ArrayList<Descriptors> descriptors = new ArrayList<Descriptors>();
    private Mat image;
    private double[] mean = new double[5];
    private double[] variance = new double[5];

    public TrainingData(String name) {
        this.name = name;
    }

    public static double roundNum(double numero, int digits) {
        int cifras = (int) Math.pow(10, digits);
        return Math.rint(numero * cifras) / cifras;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    /**
     * descriptorCase = 0 --> area
     * descriptorCase = 1 --> perimeter
     * descriptorCase = 2 --> m1
     * descriptorCase = 3 --> m2
     * descriptorCase = 4 --> m3
     *
     * @return
     */
    public double[] mahalanobisDistance(Descriptors d) {
        double result[] = new double[5];
        double dValues[] = new double[]{d.getArea(), d.getPerimeter(), d.getHuMoments()[0]
                , d.getHuMoments()[1], d.getHuMoments()[2]};
        for (int i = 0; i < descriptors.size(); i++) {
            double value = ((dValues[i] - mean[i])) * ((dValues[i] - mean[i]))
                    / variance[i];
            result[i] += roundNum(value, 2);
        }
        return result;
    }

    public void addDescriptors(Descriptors d) {
        descriptors.add(d);
    }

    public Mat getImage() {
        return image;
    }

    public void setImage(Mat image) {
        this.image = image;
    }

    /**
     * This method compute the mean of each descriptor + its variance and finally calculates the mahalanobis distance
     * of each descriptor for each item/object
     */
    public void computeCalculations() {
        /*
        MEAN CALCULATION
         */
        double mArea = 0;
        double mPerimeter = 0;
        double m1 = 0;
        double m2 = 0;
        double m3 = 0;

        for (int i = 0; i < descriptors.size(); i++) {
            mArea += descriptors.get(i).getArea();
            mPerimeter += descriptors.get(i).getPerimeter();
            m1 += descriptors.get(i).getHuMoments()[0];
            m2 += descriptors.get(i).getHuMoments()[1];
            m3 += descriptors.get(i).getHuMoments()[2];
        }
        mean[0] = mArea / descriptors.size();
        mean[1] = mPerimeter / descriptors.size();
        mean[2] = m1 / descriptors.size();
        mean[3] = m2 / descriptors.size();
        mean[4] = m3 / descriptors.size();

        /*
        VARIANCE CALCULATION
         */

        mArea = 0;
        mPerimeter = 0;
        m1 = 0;
        m2 = 0;
        m3 = 0;

        for (int i = 0; i < descriptors.size(); i++) {
            mArea += (descriptors.get(i).getArea() - mean[0]) * (descriptors.get(i).getArea() - mean[0]);
            mPerimeter += (descriptors.get(i).getPerimeter() - mean[1]) * (descriptors.get(i).getPerimeter() - mean[1]);
            m1 += (descriptors.get(i).getHuMoments()[0] - mean[2]) * (descriptors.get(i).getHuMoments()[0] - mean[2]);
            m2 += (descriptors.get(i).getHuMoments()[1] - mean[3]) * (descriptors.get(i).getHuMoments()[1] - mean[3]);
            m3 += (descriptors.get(i).getHuMoments()[2] - mean[4]) * (descriptors.get(i).getHuMoments()[2] - mean[4]);
        }

        /*variance[0] = (mArea) / descriptors.size();
        variance[1] = (mPerimeter) / descriptors.size();
        variance[2] = (m1) / descriptors.size();
        variance[3] = (m2) / descriptors.size();
        variance[4] = (m3) / descriptors.size();*/

        variance[0] = ((mean[0] * 0.01) * (mean[0] * 0.01) + mArea) / descriptors.size();
        variance[1] = ((mean[1] * 0.01) * (mean[1] * 0.01) + mPerimeter) / descriptors.size();
        variance[2] = ((mean[2] * 0.01) * (mean[2] * 0.01) + m1) / descriptors.size();
        variance[3] = ((mean[3] * 0.01) * (mean[3] * 0.01) + m2) / descriptors.size();
        variance[4] = ((mean[4] * 0.01) * (mean[4] * 0.01) + m3) / descriptors.size();

        /*
        Variance correction
        */

        /*variance[0] = (mean[0] * 0.01) * (mean[0] * .01) / descriptors.size()
                + (descriptors.size() - 1) / descriptors.size() * variance[0];
        variance[1] = (mean[1] * 0.01) * (mean[1] * .01) / descriptors.size()
                + (descriptors.size() - 1) / descriptors.size() * variance[1];
        variance[2] = (mean[2] * 0.01) * (mean[2] * .01) / descriptors.size()
                + (descriptors.size() - 1) / descriptors.size() * variance[2];
        variance[3] = (mean[3] * 0.01) * (mean[3] * .01) / descriptors.size()
                + (descriptors.size() - 1) / descriptors.size() * variance[3];
        variance[4] = (mean[4] * 0.01) * (mean[4] * .01) / descriptors.size()
                + (descriptors.size() - 1) / descriptors.size() * variance[4];*/

    }

    public ArrayList<Descriptors> getDescriptors() {
        return descriptors;
    }

    public void setDescriptors(ArrayList<Descriptors> descriptors) {
        this.descriptors = descriptors;
    }

    public double getAreaMean() {
        return mean[0];
    }

    public double getAreaVariance() {
        return variance[0];
    }

    public double getPerimeterMean() {
        return mean[1];
    }

    public double getPerimeterVariance() {
        return variance[1];
    }

    public double getM1Mean() {
        return mean[2];
    }

    public double getM1Variance() {
        return variance[2];
    }

    public double getM2Mean() {
        return mean[3];
    }

    public double getM2Variance() {
        return variance[3];
    }

    public double getM3Mean() {
        return mean[4];
    }

    public double getM3Variance() {
        return variance[4];
    }

    public void setMean(double mean, int index) {
        this.mean[index] = mean;
    }

    public void setVariance(double variance, int index) {
        this.variance[index] = variance;
    }

    public String storeData() {
        String writer = "";
        writer += (name + " ");
        for (int i = 0; i < descriptors.size(); i++) {
            writer += (mean[i] + " " + variance[i] + " ");
            Descriptors d = descriptors.get(i);
            writer += (d.getName() + " " + d.getArea() + " " + d.getPerimeter() + " ");
            for (int j = 0; j < d.getHuMoments().length; j++) {
                writer += (d.getHuMoments()[j] + " ");
            }
        }
        return writer;
    }
}
