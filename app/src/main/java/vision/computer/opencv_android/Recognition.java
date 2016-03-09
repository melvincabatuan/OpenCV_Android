package vision.computer.opencv_android;

import android.os.Environment;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

/**
 * Created by Manuel on 09/03/2016.
 */
public class Recognition {

    public Recognition(){}

    public static Mat loadImageFromFile(String fileName) {

        Mat rgbLoadedImage = null;

        File root = Environment.getExternalStorageDirectory();
        File file = new File(root, fileName);

        // this should be in BGR format according to the
        // documentation.
        Mat image = Imgcodecs.imread(file.getAbsolutePath());

        if (image.width() > 0) {

            rgbLoadedImage = new Mat(image.size(), image.type());

            Imgproc.cvtColor(image, rgbLoadedImage, Imgproc.COLOR_BGR2RGB);

            image.release();
            image = null;
        }
        return rgbLoadedImage;
    }


}
