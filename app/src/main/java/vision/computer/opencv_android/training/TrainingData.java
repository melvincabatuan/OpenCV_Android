package vision.computer.opencv_android.training;

import org.opencv.core.Mat;

import java.util.ArrayList;

/**
 * Created by Manuel on 14/03/2016.
 */
public class TrainingData {

    private ArrayList<Descriptors> descriptors = new ArrayList<Descriptors>();
    private Mat image;
    private double areaMean, areaVariance, areaDistance;
    private double perimeterMean, perimeterVariance, perimeterDistance;
    private double m1Mean, m1Variance, m1Distance;
    private double m2Mean, m2Variance, m2Distance;
    private double m3Mean, m3Variance, m3Distance;

    public TrainingData() {
    }

    /**
     * descriptorCase = 1 --> area
     * descriptorCase = 2 --> perimeter
     * descriptorCase = 3 --> m1
     * descriptorCase = 4 --> m2
     * descriptorCase = 5 --> m3
     *
     * @param descriptorCase
     * @return
     */
    private double mahalanobisDistance(int descriptorCase) {
        double result = 0;
        switch (descriptorCase) {
            case 1:
                for (int i = 0; i < descriptors.size(); i++) {
                    result += ((descriptors.get(i).getArea() - areaMean)) * ((descriptors.get(i).getArea() - areaMean))
                            / areaVariance;
                }
                return result;
            case 2:
                for (int i = 0; i < descriptors.size(); i++) {
                    result += ((descriptors.get(i).getPerimeter() - perimeterMean)) * ((descriptors.get(i).getPerimeter() - perimeterMean))
                            / perimeterVariance;
                }
                return result;
            case 3:
                for (int i = 0; i < descriptors.size(); i++) {
                    result += ((descriptors.get(i).getHuMoments()[0] - m1Mean)) * ((descriptors.get(i).getHuMoments()[0] - m1Mean))
                            / m1Variance;
                }
                return result;
            case 4:
                for (int i = 0; i < descriptors.size(); i++) {
                    result += ((descriptors.get(i).getHuMoments()[1] - m2Mean)) * ((descriptors.get(i).getHuMoments()[1] - m2Mean))
                            / m2Variance;
                }
                return result;
            case 5:
                for (int i = 0; i < descriptors.size(); i++) {
                    result += ((descriptors.get(i).getHuMoments()[2] - m3Mean)) * ((descriptors.get(i).getHuMoments()[2] - m3Mean))
                            / m3Variance;
                }
                return result;
        }
        return 0;
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

        areaDistance = mahalanobisDistance(0);
        perimeterDistance = mahalanobisDistance(1);
        m1Distance = mahalanobisDistance(2);
        m2Distance = mahalanobisDistance(3);
        m3Distance = mahalanobisDistance(4);
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

    public double getAreaDistance() {
        return areaDistance;
    }

    public double getPerimeterDistance() {
        return perimeterDistance;
    }

    public double getM1Distance() {
        return m1Distance;
    }

    public double getM2Distance() {
        return m2Distance;
    }

    public double getM3Distance() {
        return m3Distance;
    }
}
