package vision.computer.opencv_android.training;

import org.opencv.core.Mat;

/**
 * Created by Manuel on 14/03/2016.
 */
public class Descriptors {

    private double area;
    private double perimeter;
    private double[] huMoments;

    private String name;

    public Descriptors(){}

    public Descriptors(String name, double area, double perimeter, Mat huMoments) {
        this.name = name;
        this.area = area;
        this.perimeter = perimeter;
        this.huMoments = huMoments.get(0,0);
    }

    public double getArea() {
        return area;
    }

    public void setArea(double area) {
        this.area = area;
    }

    public double getPerimeter() {
        return perimeter;
    }

    public void setPerimeter(double perimeter) {
        this.perimeter = perimeter;
    }

    public double[] getHuMoments() {
        return huMoments;
    }

    public void setHuMoments(double[] huMoments) {
        this.huMoments = huMoments;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }
}
