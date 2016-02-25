package vision.computer.opencv_android;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Context;
import android.content.Intent;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Parameters;
import android.hardware.Camera.Size;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.provider.MediaStore.Images;
import android.support.v7.app.ActionBarActivity;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.SubMenu;
import android.view.Window;
import android.view.WindowManager;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

// Use the deprecated Camera class.
@SuppressWarnings("deprecation")

public class MainActivity extends ActionBarActivity
        implements CvCameraViewListener2 {
    // A tag for log output.
    private static final String TAG =
            MainActivity.class.getSimpleName();

    // A key for storing the index of the active camera.
    private static final String STATE_CAMERA_INDEX = "cameraIndex";

    // A key for storing the index of the active image size.
    private static final String STATE_IMAGE_SIZE_INDEX =
            "imageSizeIndex";

    // An ID for items in the image size submenu.
    private static final int MENU_GROUP_ID_SIZE = 3;
    private static final int MENU_GROUP_ID_TYPE = 2;
    private static final int SMENU_ADJUST = 4;
    ArrayList<String> imageTypes = new ArrayList<String>();
    // The index of the active camera.
    private int mCameraIndex;
    // The index of the active image size.
    private int mImageSizeIndex;
    // The image sizes supported by the active camera.
    private List<Size> mSupportedImageSizes;
    // The camera view.
    private CameraBridgeViewBase mCameraView;
    // Whether the next camera frame should be saved as a photo.
    private boolean mIsPhotoPending;
    // A matrix that is used when saving photos.
    private Mat mBgr;
    // Whether an asynchronous menu action is in progress.
    // If so, menu interaction should be disabled.
    private boolean mIsMenuLocked;
    // The OpenCV loader callback.

    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;

    private BaseLoaderCallback mLoaderCallback =
            new BaseLoaderCallback(this) {
                @Override
                public void onManagerConnected(final int status) {
                    switch (status) {
                        case LoaderCallbackInterface.SUCCESS:
                            Log.d(TAG, "OpenCV loaded successfully");
                            initializeOpenCVDependencies();
                            break;
                        default:
                            super.onManagerConnected(status);
                            break;
                    }
                }
            };
    /*  mPhotoType = 0 --> Normal camera
        mPhotoType = 1 --> CLAHE algorithm
        mPhotoType = 2 --> equalize hist
        mPhotoType = 3 --> Alien effect
        mPhotoType = 4 --> Poster effect
        mPhotoType = 5 --> Distorsion effect */
    private int mPhotoType = 0;

    private int mAdjustLevel = -1;

    private void initializeOpenCVDependencies(){
        try{
            // Copy the resource into a temp file so OpenCV can load it
            InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
            FileOutputStream os = new FileOutputStream(mCascadeFile);

            byte[] buffer = new byte[4096];
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                os.write(buffer, 0, bytesRead);
            }
            is.close();
            os.close();

            // Load the cascade classifier
            //cascadeClassifier = new CascadeClassifier(mCascadeFile.getAbsolutePath());
            cascadeClassifier.load(mCascadeFile.getAbsolutePath());
            if(cascadeClassifier.empty()){
                throw new RuntimeException("CASCADE EMPTY");
            }
        } catch (Exception e) {
            Log.e("OpenCVActivity", "Error loading cascade", e);
        }

        // And we are ready to go
        mCameraView.enableView();
        mBgr = new Mat();
    }
    // Suppress backward incompatibility errors because we provide
    // backward-compatible fallbacks.
    @SuppressLint("NewApi")
    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        imageTypes.add(getResources().getString(R.string.menu_normal));
        imageTypes.add(getResources().getString(R.string.menu_clahe));
        imageTypes.add(getResources().getString(R.string.menu_heist));
        imageTypes.add(getResources().getString(R.string.menu_alien));
        imageTypes.add(getResources().getString(R.string.menu_poster));
        imageTypes.add(getResources().getString(R.string.menu_distorsionB));
        imageTypes.add(getResources().getString(R.string.menu_distorsionC));

        final Window window = getWindow();
        window.addFlags(
                WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        if (savedInstanceState != null) {
            mCameraIndex = savedInstanceState.getInt(
                    STATE_CAMERA_INDEX, 0);
            mImageSizeIndex = savedInstanceState.getInt(
                    STATE_IMAGE_SIZE_INDEX, 0);
        } else {
            mCameraIndex = 0;
            mImageSizeIndex = 0;
        }

        final Camera camera;

        CameraInfo cameraInfo = new CameraInfo();
        Camera.getCameraInfo(mCameraIndex, cameraInfo);

        camera = Camera.open(mCameraIndex);

        final Parameters parameters = camera.getParameters();
        camera.release();
        mSupportedImageSizes =
                parameters.getSupportedPreviewSizes();
        final Size size = mSupportedImageSizes.get(mImageSizeIndex);

        mCameraView = new JavaCameraView(this, mCameraIndex);
        mCameraView.setMaxFrameSize(size.width, size.height);
        mCameraView.setCvCameraViewListener(this);
        setContentView(mCameraView);
    }

    public void onSaveInstanceState(Bundle savedInstanceState) {
        // Save the current camera index.
        savedInstanceState.putInt(STATE_CAMERA_INDEX, mCameraIndex);

        // Save the current image size index.
        savedInstanceState.putInt(STATE_IMAGE_SIZE_INDEX,
                mImageSizeIndex);

        super.onSaveInstanceState(savedInstanceState);
    }

    // Suppress backward incompatibility errors because we provide
    // backward-compatible fallbacks.
    @SuppressLint("NewApi")
    @Override
    public void recreate() {
        if (Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.HONEYCOMB) {
            super.recreate();
        } else {
            finish();
            startActivity(getIntent());
        }
    }

    @Override
    public void onPause() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        super.onPause();
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0,
                this, mLoaderCallback);
        mIsMenuLocked = false;
    }

    @Override
    public void onDestroy() {
        if (mCameraView != null) {
            mCameraView.disableView();
        }
        super.onDestroy();
    }

    @Override
    public boolean onCreateOptionsMenu(final Menu menu) {
        getMenuInflater().inflate(R.menu.activity_main, menu);

        if (mSupportedImageSizes.size() > 1) {
            final SubMenu sizeSubMenu = menu.addSubMenu(
                    R.string.menu_image_size);
            //Add all the sizes (resolution) that teh camera offers to the sub Menu
            for (int i = 0; i < mSupportedImageSizes.size(); i++) {
                final Size size = mSupportedImageSizes.get(i);
                sizeSubMenu.add(MENU_GROUP_ID_SIZE, i, Menu.NONE,
                        String.format("%dx%d", size.width,
                                size.height));
            }
        }
        final SubMenu imageSubMenu = menu.addSubMenu(
                R.string.menu_image_type);
        int i = 0;
        for (String s : imageTypes) {
            imageSubMenu.add(MENU_GROUP_ID_TYPE, i, Menu.NONE, s);
            i++;
        }
        //SubMenu adjust = imageSubMenu.addSubMenu("Adjust");
        /*final SubMenu barrilSubMenu = menu.addSubMenu(
                R.string.menu_distorsionB);
        for(i = 0; i< 10; i++)
            barrilSubMenu.add(SMENU_ADJUST,i,Menu.NONE,i);

        final SubMenu cojinSubMenu = menu.addSubMenu(
                R.string.menu_distorsionC);
        for(i = 0; i< 10; i++)
            cojinSubMenu.add(SMENU_ADJUST,i,Menu.NONE,i);

        final SubMenu CLAHESubMenu = menu.addSubMenu(
                R.string.menu_clahe);
        for(i = 0; i< 10; i++)
            CLAHESubMenu.add(SMENU_ADJUST,i,Menu.NONE,i);*/
        return true;
    }

    // Suppress backward incompatibility errors because we provide
    // backward-compatible fallbacks (for recreate).
    @SuppressLint("NewApi")
    @Override
    public boolean onOptionsItemSelected(final MenuItem item) {
        if (mIsMenuLocked) {
            return true;
        }
        if (item.getGroupId() == MENU_GROUP_ID_SIZE) {
            //Update of the camera resolution and frame re-creation
            mImageSizeIndex = item.getItemId();
            recreate();
            return true;
        }
        if (item.getGroupId() == MENU_GROUP_ID_TYPE) {
            if (item.getTitle().equals(getResources().getString(R.string.menu_normal)))
                mPhotoType = 0;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_clahe)))
                mPhotoType = 1;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_heist)))
                mPhotoType = 2;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_alien)))
                mPhotoType = 3;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_poster)))
                mPhotoType = 4;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionB)))
                mPhotoType = 5;
            else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionC)))
                mPhotoType = 6;

            return true;
        }
        /*if(item.getGroupId() == SMENU_ADJUST){
            mAdjustLevel = item.getItemId();
            return true;
        }*/
        switch (item.getItemId()) {
            case R.id.menu_take_photo:
                //Do not let to use the menu while taking a photo
                mIsMenuLocked = true;
                // FLAG Next frame, take the photo.
                mIsPhotoPending = true;

                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    @Override
    public void onCameraViewStarted(final int width,
                                    final int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);
        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(final CvCameraViewFrame inputFrame) {
        final Mat rgba = inputFrame.rgba();

        switch (mPhotoType) {
            case 0:
                //NORMAL PHOTO
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
                break;
            case 1:
                //CLAHE - Contrast Limited AHE
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
                mBgr = clahe(mBgr, 2);
                break;
            case 2:
                //HISTOGRAM EQUALIZATION
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
                mBgr = histEqual(mBgr);
                break;
            case 3:
                //ALIEN EFFECT
                //Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                //mBgr = alien(mBgr);
            case 4:
                //POSTER EFFECT
                //Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                //mBgr = poster(mBgr);
            case 5:
                //DISTORSION BARRIL EFFECT
                //Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                //mBgr = distorsionBarril(mBgr,-1);
            case 6:
                //DISTORSION COJIN EFFECT
                //Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                //mBgr = distorsionCojin(mBgr,-1);
        }
        Imgproc.cvtColor(mBgr,rgba,Imgproc.COLOR_BGR2RGBA);

        if (mIsPhotoPending) {
            mIsPhotoPending = false;
            takePhoto();
        }
        return rgba;
    }

    private Mat alien(Mat bgr){
        Imgproc.cvtColor(bgr, grayscaleImage, Imgproc.COLOR_RGBA2RGB);
        MatOfRect faces = new MatOfRect();


        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2, new org.opencv.core.Size(absoluteFaceSize, absoluteFaceSize), new org.opencv.core.Size());
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i <facesArray.length; i++)
            Imgproc.rectangle(bgr, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        return bgr;
    }

    public boolean R1(int R, int G, int B) {
        boolean e1 = (R>95) && (G>40) && (B>20) && ((Math.max(R,Math.max(G,B)) - Math.min(R, Math.min(G,B)))>15) && (Math.abs(R - G)>15) && (R>G) && (R>B);
        boolean e2 = (R>220) && (G>210) && (B>170) && (Math.abs(R - G)<=15) && (R>B) && (G>B);
        return (e1||e2);
    }

    public boolean R2(float Y, float Cr, float Cb) {
        boolean e3 = Cr <= 1.5862*Cb+20;
        boolean e4 = Cr >= 0.3448*Cb+76.2069;
        boolean e5 = Cr >= -4.5652*Cb+234.5652;
        boolean e6 = Cr <= -1.15*Cb+301.75;
        boolean e7 = Cr <= -2.2857*Cb+432.85;
        return e3 && e4 && e5 && e6 && e7;
    }

    boolean R3(float H, float S, float V) {
        return (H<25) || (H > 230);
    }

    public Mat getSkin(Mat src) {
        // allocate the result matrix
        Mat dst = src.clone();

        byte[] cwhite = new byte[4];
        cwhite[0]=Byte.MAX_VALUE;
        cwhite[1]=Byte.MAX_VALUE;
        cwhite[2]=Byte.MAX_VALUE;
        cwhite[3]=Byte.MAX_VALUE;
        byte[] cblack = new byte[4];
        cblack[0]=Byte.MIN_VALUE;
        cblack[1]=Byte.MIN_VALUE;
        cblack[2]=Byte.MIN_VALUE;
        cblack[3]=Byte.MIN_VALUE;


        Mat src_ycrcb= new Mat(), src_hsv = new Mat();
        // OpenCV scales the YCrCb components, so that they
        // cover the whole value range of [0,255], so there's
        // no need to scale the values:
        Imgproc.cvtColor(src, src_ycrcb, Imgproc.COLOR_BGR2YCrCb);
        // OpenCV scales the Hue Channel to [0,180] for
        // 8bit images, so make sure we are operating on
        // the full spectrum from [0,360] by using floating
        // point precision:
        src.convertTo(src_hsv, CvType.CV_32FC3);
        Imgproc.cvtColor(src_hsv, src_hsv,Imgproc.COLOR_BGR2HSV);
        // Now scale the values between [0,255]:
        Core.normalize(src_hsv, src_hsv, 0.0, 255.0, Core.NORM_MINMAX, CvType.CV_32FC3);

        for(int i = 0; i < src.rows(); i++) {
            for(int j = 0; j < src.cols(); j++) {
                double[] pix_bgr = src.get(i,j);
                double B = pix_bgr[0];
                double G = pix_bgr[1];
                double R = pix_bgr[2];
                // apply rgb rules
                boolean a = R1((int) R, (int) G, (int) B);

                double[] pix_ycrcb = src_ycrcb.get(i,j);
                double Y = pix_ycrcb[0];
                double Cr = pix_ycrcb[1];
                double Cb = pix_ycrcb[2];
                // apply ycrcb rule
                boolean b = R2((int)Y,(int)Cr,(int)Cb);

                double[] pix_hsv = src_hsv.get(i,j);
                float H = (float) pix_hsv[0];
                float S = (float) pix_hsv[1];
                float V = (float) pix_hsv[2];
                // apply hsv rule
                boolean c = R3(H,S,V);

                if(!(a&&b&&c))
                    dst.put(i,j,cblack);
            }
        }
        return dst;
    }
    private Mat poster(Mat bgr){
        return null;
    }

    private Mat distorsionCojin(Mat bgr, int adjust){
        return null;
    }

    private Mat distorsionBarril(Mat bgr, int adjust){
        return null;
    }

    /**
     * This functions implements a histogram equalization with a limit in the contrast.
     * We have to use the color space Lab (L for light, a and b for the colours)
     * in order to use CLAHE algorithm. the Algorithm will be applied to the channel L
     * and the result will be merged with the rest of the colours of the image.
     *
     * @param bgr
     * @param limit
     * @return
     */
    private Mat clahe(Mat bgr, int limit) {
        if (bgr.channels() >= 3) {
            Mat labImg = new Mat();
            List<Mat> channels = new ArrayList<Mat>();
            CLAHE cl = Imgproc.createCLAHE();
            cl.setClipLimit(limit);

            Imgproc.cvtColor(bgr, labImg, Imgproc.COLOR_BGR2Lab);

            Core.split(labImg, channels);
            //Apply on the channel L (Light)
            cl.apply(channels.get(0), channels.get(0));
            Core.merge(channels, labImg);

            Imgproc.cvtColor(labImg, bgr, Imgproc.COLOR_Lab2BGR);

            return bgr;
        } else
            return null;

    }

    /**
     * This function equalize the histogram of a photo and return it. In BGR format.
     * It change its format to HSV, split in three channels equalize V, merge them
     * and re-format to BGR
     *
     * @param bgr
     * @return BGR equalized histogram
     */
    private Mat histEqual(Mat bgr) {
        if (bgr.channels() >= 3) {
            Mat aux = new Mat();
            Mat heistMat = new Mat();
            List<Mat> channels = new ArrayList<Mat>();

            Imgproc.cvtColor(bgr, aux, Imgproc.COLOR_BGR2YCrCb);
            Core.split(aux, channels);
            //Get channel Y, the one that represents the gray scale of the image
            // we are going to equalize its histogram.
            Imgproc.equalizeHist(channels.get(0), channels.get(0));
            Core.merge(channels, aux);
            Imgproc.cvtColor(aux, heistMat, Imgproc.COLOR_YCrCb2BGR);

            return heistMat;
        } else
            return null;
    }

    private void takePhoto() {

        // Determine the path and metadata for the photo.
        final long currentTimeMillis = System.currentTimeMillis();
        final String appName = getString(R.string.app_name);
        final String galleryPath =
                Environment.getExternalStoragePublicDirectory(
                        Environment.DIRECTORY_PICTURES).toString();
        final String albumPath = galleryPath + File.separator +
                appName;
        final String photoPath = albumPath + File.separator +
                currentTimeMillis + Image_activity.PHOTO_FILE_EXTENSION;
        final ContentValues values = new ContentValues();

        values.put(MediaStore.MediaColumns.DATA, photoPath);
        values.put(Images.Media.MIME_TYPE,
                Image_activity.PHOTO_MIME_TYPE);
        values.put(Images.Media.TITLE, appName);
        values.put(Images.Media.DESCRIPTION, appName);
        values.put(Images.Media.DATE_TAKEN, currentTimeMillis);

        // Ensure that the album directory exists.
        File album = new File(albumPath);
        if (!album.isDirectory() && !album.mkdirs()) {
            Log.e(TAG, "Failed to create album directory at " +
                    albumPath);
            onTakePhotoFailed();
            return;
        }

        if (!Imgcodecs.imwrite(photoPath, mBgr)) {
            Log.e(TAG, "Failed to save photo to " + photoPath);
            onTakePhotoFailed();
        }
        Log.d(TAG, "Photo saved successfully to " + photoPath);

        // Try to insert the photo into the MediaStore.
        Uri uri;
        try {
            uri = getContentResolver().insert(
                    MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
        } catch (final Exception e) {
            Log.e(TAG, "Failed to insert photo into MediaStore");
            e.printStackTrace();

            // Since the insertion failed, delete the photo.
            File photo = new File(photoPath);
            if (!photo.delete()) {
                Log.e(TAG, "Failed to delete non-inserted photo");
            }

            onTakePhotoFailed();
            return;
        }

        // Open the photo in LabActivity.
        final Intent intent = new Intent(this, Image_activity.class);
        intent.putExtra(Image_activity.EXTRA_PHOTO_URI, uri);
        intent.putExtra(Image_activity.EXTRA_PHOTO_DATA_PATH,
                photoPath);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                startActivity(intent);
            }
        });
    }

    private void onTakePhotoFailed() {
        mIsMenuLocked = false;

        // Show an error message.
        final String errorMessage =
                getString(R.string.photo_error_message);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(MainActivity.this, errorMessage,
                        Toast.LENGTH_SHORT).show();
            }
        });
    }
}