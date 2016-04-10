package vision.computer.opencv_android.contours;

import org.opencv.core.Point;

/**
 * Created by Fernando on 10/04/2016.
 */
public class Line {

    private Point point1;
    private Point point2;

    public Line(Point point1, Point point2) {
        this.point1 = point1;
        this.point2 = point2;
    }
}
