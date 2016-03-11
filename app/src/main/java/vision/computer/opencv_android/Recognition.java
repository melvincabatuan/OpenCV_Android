package vision.computer.opencv_android;

import android.os.Environment;
import android.support.design.widget.Snackbar;
import android.view.View;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;

import java.io.File;

/**
 * Created by Manuel on 09/03/2016.
 */
public class Recognition {

    private static  View view;

    public Recognition(View view) {
        this.view = view;
    }

    public static Mat loadImageFromFile(String fileName, int width, int height) {

        if (android.os.Environment.getExternalStorageState().equals(
                android.os.Environment.MEDIA_MOUNTED)) {

            Mat image = new Mat(new Size(width, height), CvType.CV_8U);
            File roo=Environment.getDataDirectory();
            File ro=Environment.getDownloadCacheDirectory();
            File roott=Environment.getExternalStorageDirectory();
            File roottt=Environment.getRootDirectory();
            File root = Environment.getExternalStorageDirectory();
            String path = "/storage/sdcard1/";
            File file = new File(path+fileName);

            // this should be in BGR format according to the
            // documentation.
            image = Imgcodecs.imread(file.getAbsolutePath());

            if (image.width() > 0) {
                Snackbar snackbar;
                snackbar = Snackbar.make(view, "I'm innnnnnn ", Snackbar.LENGTH_LONG);
                snackbar.show();
            }
            return image;
        }
        else return null;
    }


}
