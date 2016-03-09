package vision.computer.opencv_android;

import android.os.Environment;
import android.util.Log;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;

/**
 * Created by Manuel on 09/03/2016.
 */
public class Recognition {

    public Recognition() {
    }

    public static Mat loadImageFromFile(String fileName, int width, int height) {

        Mat image = new Mat(new Size(width, height), CvType.CV_8U);// Change CvType as you need.

        File path = Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES);
        File file = new File(path, fileName);

        // this should be in BGR format according to the
        // documentation.
        image = Imgcodecs.imread(file.getAbsolutePath());

        Mat bgr = new Mat(image.size(), image.type());

        Log.d("TAG", file.getAbsolutePath() + "  PATH     --------");
        if (image.width() > 0) {

            bgr = new Mat(image.size(), image.type());

            Imgproc.cvtColor(image, bgr, Imgproc.COLOR_BGR2RGB);

            image.release();
            image = null;
        }
        return bgr;
    }


}
