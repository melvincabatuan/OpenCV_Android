package vision.computer.opencv_android;

import android.annotation.SuppressLint;
import android.content.ContentValues;
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
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;

import java.io.File;
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
    private BaseLoaderCallback mLoaderCallback =
            new BaseLoaderCallback(this) {
                @Override
                public void onManagerConnected(final int status) {
                    switch (status) {
                        case LoaderCallbackInterface.SUCCESS:
                            Log.d(TAG, "OpenCV loaded successfully");
                            mCameraView.enableView();
                            mBgr = new Mat();
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

        final SubMenu barrilSubMenu = menu.addSubMenu(
                R.string.menu_distorsionB);
        for(i = 0; i< 10; i++)
            barrilSubMenu.add(i);

        final SubMenu cojinSubMenu = menu.addSubMenu(
                R.string.menu_distorsionC);
        for(i = 0; i< 10; i++)
            cojinSubMenu.add(i);

        final SubMenu CLAHESubMenu = menu.addSubMenu(
                R.string.menu_clahe);
        for(i = 0; i< 10; i++)
            CLAHESubMenu.add(i);
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
            if (item.getTitle().equals(getResources().getString(R.string.menu_clahe)))
                mPhotoType = 1;
            if (item.getTitle().equals(getResources().getString(R.string.menu_heist)))
                mPhotoType = 2;
            if (item.getTitle().equals(getResources().getString(R.string.menu_alien)))
                mPhotoType = 3;
            if (item.getTitle().equals(getResources().getString(R.string.menu_poster)))
                mPhotoType = 4;
            if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionB)))
                mPhotoType = 5;
            if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionC)))
                mPhotoType = 6;
            return true;
        }
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
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(final CvCameraViewFrame inputFrame) {
        final Mat rgba = inputFrame.rgba();

        if (mIsPhotoPending) {
            mIsPhotoPending = false;
            takePhoto(rgba);
        }
        return rgba;

    }

    private Mat alien(Mat bgr){
        return null;
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
            //Apply on the channel L
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
            Imgproc.equalizeHist(channels.get(0), channels.get(0));
            Core.merge(channels, aux);
            Imgproc.cvtColor(aux, heistMat, Imgproc.COLOR_YCrCb2BGR, 3);

            return aux;
        } else
            return null;
    }

    private void takePhoto(final Mat rgba) {

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

        // Try to create the photo.
        switch (mPhotoType) {
            case 0:
                //NORMAL PHOTO
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
            case 1:
                //CLAHE - Contrast Limited AHE
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);
                mBgr = clahe(mBgr, 2);
            case 2:
                //HISTOGRAM EQUALIZATION
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                mBgr = histEqual(mBgr);
            case 3:
                //ALIEN EFFECT
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                mBgr = alien(mBgr);
            case 4:
                //POSTER EFFECT
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                mBgr = poster(mBgr);
            case 5:
                //DISTORSION BARRIL EFFECT
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                mBgr = distorsionBarril(mBgr,-1);
            case 6:
                //DISTORSION COJIN EFFECT
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR);
                mBgr = distorsionCojin(mBgr,-1);
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