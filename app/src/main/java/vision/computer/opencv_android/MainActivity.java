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
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
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
    private ArrayList<String> mPhotoType = new ArrayList<String>();

    private int mAdjustLevel = -1;

    private void initializeOpenCVDependencies() {
        try {
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
            cascadeClassifier = new CascadeClassifier("android.resource://OpenCV_Android/app/src/main/raw/haarcascade_frontalface_alt.xml");
            if (cascadeClassifier.empty()) {
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
        imageTypes.add(getResources().getString(R.string.menu_alienHSV));
        imageTypes.add(getResources().getString(R.string.menu_poster));
        imageTypes.add(getResources().getString(R.string.menu_posterContrast));
        imageTypes.add(getResources().getString(R.string.menu_distorsionB));
        imageTypes.add(getResources().getString(R.string.menu_distorsionC));
        imageTypes.add(getResources().getString(R.string.menu_sepia));

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
                if (mPhotoType.contains(getResources().getString(R.string.menu_normal)))
                    mPhotoType.clear();
                else
                    mPhotoType.add(getResources().getString(R.string.menu_normal));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_clahe)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_clahe)))
                    mPhotoType.remove(getResources().getString(R.string.menu_clahe));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_clahe));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_heist)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_clahe)))
                    mPhotoType.remove(getResources().getString(R.string.menu_clahe));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_clahe));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_alien)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_alien)))
                    mPhotoType.remove(getResources().getString(R.string.menu_alien));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_alien));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_alienHSV)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_alienHSV)))
                    mPhotoType.remove(getResources().getString(R.string.menu_alienHSV));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_alienHSV));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_poster)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_poster)))
                    mPhotoType.remove(getResources().getString(R.string.menu_poster));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_poster));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_posterContrast)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_posterContrast)))
                    mPhotoType.remove(getResources().getString(R.string.menu_posterContrast));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_posterContrast));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionB)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_distorsionB)))
                    mPhotoType.remove(getResources().getString(R.string.menu_distorsionB));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_distorsionB));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionC)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_distorsionC)))
                    mPhotoType.remove(getResources().getString(R.string.menu_distorsionC));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_distorsionC));

            else if (item.getTitle().equals(getResources().getString(R.string.menu_sepia)))
                if (mPhotoType.contains(getResources().getString(R.string.menu_sepia)))
                    mPhotoType.remove(getResources().getString(R.string.menu_sepia));
                else
                    mPhotoType.add(getResources().getString(R.string.menu_sepia));

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
        boolean post = true;
        Filters F = new Filters();
        Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);

        if (mPhotoType.contains(getResources().getString(R.string.menu_clahe)))
            mBgr = F.clahe(mBgr, 2);

        if (mPhotoType.contains(getResources().getString(R.string.menu_heist)))
            mBgr = F.histEqual(mBgr);

        if (mPhotoType.contains(getResources().getString(R.string.menu_alien)))
            mBgr = F.getSkin(rgba);

        if (mPhotoType.contains(getResources().getString(R.string.menu_alienHSV))){
            mBgr = F.alienHSV(mBgr);
            post = false;
        }
        if (mPhotoType.contains(getResources().getString(R.string.menu_poster)))
            mBgr = F.poster(mBgr, 10);

        if (mPhotoType.contains(getResources().getString(R.string.menu_posterContrast)))
            mBgr = F.poster2(mBgr);

        if (mPhotoType.contains(getResources().getString(R.string.menu_distorsionB)))
            mBgr = F.distorsionBarril(mBgr, -1);

        if (mPhotoType.contains(getResources().getString(R.string.menu_distorsionC)))
            mBgr = F.distorsionBarril(mBgr, 1);

        if (mPhotoType.contains(getResources().getString(R.string.menu_sepia)))
            mBgr = F.sepia(mBgr, 1);

        if (post) {
            Imgproc.cvtColor(mBgr, rgba, Imgproc.COLOR_BGR2RGBA, 3);
        } else {
            return mBgr;
        }

        if (mIsPhotoPending) {
            mIsPhotoPending = false;
            takePhoto();
        }
        return rgba;
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