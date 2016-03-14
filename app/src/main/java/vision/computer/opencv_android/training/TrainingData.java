package vision.computer.opencv_android.training;

import java.util.ArrayList;

/**
 * Created by Manuel on 14/03/2016.
 */
public class TrainingData {

    private ArrayList<Descriptors> descriptors = new ArrayList<Descriptors>();
    private double areaMean, areaVariance;
    private double perimeterMean, perimeterVariance;
    private double m1Mean, m1Variance;
    private double m2Mean, m2Variance;
    private double m3Mean, m3Variance;

    public TrainingData() {
    }

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
        areaMean = mArea / descriptors.size();
        perimeterMean = mPerimeter / descriptors.size();
        m1Mean = m1 / descriptors.size();
        m2Mean = m2 / descriptors.size();
        m3Mean = m3 / descriptors.size();

        /*
        VARIANCE CALCULATION
         */
        mArea = 0;
        mPerimeter = 0;
        m1 = 0;
        m2 = 0;
        m3 = 0;

        for (int i = 0; i < descriptors.size(); i++) {
            mArea += (descriptors.get(i).getArea() - areaMean) * (descriptors.get(i).getArea() - areaMean);
            mPerimeter += (descriptors.get(i).getPerimeter() - perimeterMean) * (descriptors.get(i).getPerimeter() - perimeterMean);
            m1 += (descriptors.get(i).getHuMoments()[0] - m1Mean) * (descriptors.get(i).getHuMoments()[0] - m1Mean);
            m2 += (descriptors.get(i).getHuMoments()[1] - m2Mean) * (descriptors.get(i).getHuMoments()[1] - m2Mean);
            m3 += (descriptors.get(i).getHuMoments()[2] - m3Mean) * (descriptors.get(i).getHuMoments()[2] - m3Mean);
        }

        areaVariance = mArea / descriptors.size();
        perimeterVariance = mPerimeter / descriptors.size();
        m1Variance = m1 / descriptors.size();
        m2Variance = m2 / descriptors.size();
        m3Variance = m3 / descriptors.size();
    }

    public ArrayList<Descriptors> getDescriptors() {
        return descriptors;
    }

    public double getAreaMean() {
        return areaMean;
    }

    public double getAreaVariance() {
        return areaVariance;
    }

    public double getPerimeterMean() {
        return perimeterMean;
    }

    public double getPerimeterVariance() {
        return perimeterVariance;
    }

    public double getM1Mean() {
        return m1Mean;
    }

    public double getM1Variance() {
        return m1Variance;
    }

    public double getM2Mean() {
        return m2Mean;
    }

    public double getM2Variance() {
        return m2Variance;
    }

    public double getM3Mean() {
        return m3Mean;
    }

    public double getM3Variance() {
        return m3Variance;
    }
}
