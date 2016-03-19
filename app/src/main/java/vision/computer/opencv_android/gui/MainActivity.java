package vision.computer.opencv_android.gui;

import android.annotation.SuppressLint;
import android.content.ContentValues;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
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
import android.support.design.widget.Snackbar;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
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
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import vision.computer.opencv_android.R;
import vision.computer.opencv_android.Recognition;
import vision.computer.opencv_android.effects.Filters;

// Use the deprecated Camera class.
@SuppressWarnings("deprecation")

public class MainActivity extends AppCompatActivity
        implements CvCameraViewListener2 {

    // A tag for log output.
    private static final String TAG =
            MainActivity.class.getSimpleName();

    private static final String SD_PATH = "/Project2/";

    // A key for storing the index of the active camera.
    private static final String STATE_CAMERA_INDEX = "cameraIndex";

    // A key for storing the index of the active image size.
    private static final String STATE_IMAGE_SIZE_INDEX =
            "imageSizeIndex";

    // An ID for items in the image size submenu.
    private static final int MENU_GROUP_ID_SIZE = 3;
    private static final int MENU_GROUP_ID_TYPE = 2;
    private static final int MENU_GROUP_ID_SEG = 4;
    private static Recognition rec;
    ArrayList<String> imageTypes = new ArrayList<String>();
    String[] imageSegmentation;
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
    private CascadeClassifier cascadeClassifier;
    private Mat grayscaleImage;
    private int absoluteFaceSize;
    private Boolean mStaticImage = false;
    private double[][] resultClas;
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
    private Map<String, Integer> mPhotoType = new HashMap<String, Integer>();
    private ArrayList<String> mPhotoSeg = new ArrayList<String>();
    private String mSelectionValue = "";
    private int mLoadNext = 0;

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
    // backward-compatible fallbacks
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
        imageTypes.add(getResources().getString(R.string.menu_poster_Ellipse));
        imageTypes.add(getResources().getString(R.string.menu_posterContrast));
        imageTypes.add(getResources().getString(R.string.menu_distorsionB));
        imageTypes.add(getResources().getString(R.string.menu_distorsionC));
        imageTypes.add(getResources().getString(R.string.menu_sepia));
        imageTypes.add(getResources().getString(R.string.menu_smooth));
        imageTypes.add(getResources().getString(R.string.menu_gaussian));
        imageTypes.add(getResources().getString(R.string.menu_median));
        imageTypes.add(getResources().getString(R.string.menu_bordeado));
        imageTypes.add(getResources().getString(R.string.menu_cartoon));
        imageTypes.add(getResources().getString(R.string.menu_cartoon_v1));
        imageTypes.add(getResources().getString(R.string.menu_cartoon_v2));

        imageSegmentation = getResources().getStringArray(R.array.menu_segmentation);

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

        rec = new Recognition(findViewById(android.R.id.content), SD_PATH);

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

        final SubMenu segSubMenu = menu.addSubMenu(
                R.string.menu_image_seg);
        i = 0;

        for (String s : imageSegmentation) {
            segSubMenu.add(MENU_GROUP_ID_SEG, i, Menu.NONE, s);
            i++;
        }

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
            if (mPhotoSeg.isEmpty()) {
                mImageSizeIndex = item.getItemId();
                recreate();
            }
            return true;
        }
        if (item.getGroupId() == MENU_GROUP_ID_TYPE) {
            mStaticImage = false;
            mSelectionValue = (String) item.getTitle();
            mPhotoSeg.clear();
            if (item.getTitle().equals(getResources().getString(R.string.menu_normal)))
                mPhotoType.clear();

            else if (item.getTitle().equals(getResources().getString(R.string.menu_clahe))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_clahe)))
                    mPhotoType.put(getResources().getString(R.string.menu_clahe), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_heist))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_clahe)))
                    mPhotoType.put(getResources().getString(R.string.menu_clahe), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_alien))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_alien)))
                    mPhotoType.put(getResources().getString(R.string.menu_alien), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_alienHSV))) {
                if (mPhotoType.containsKey(getResources().getString(R.string.menu_alienHSV)))
                    mPhotoType.put(getResources().getString(R.string.menu_alienHSV), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_poster))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_poster)))
                    mPhotoType.put(getResources().getString(R.string.menu_poster), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_poster_Ellipse))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_poster_Ellipse)))
                    mPhotoType.put(getResources().getString(R.string.menu_poster_Ellipse), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_posterContrast))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_posterContrast)))
                    mPhotoType.put(getResources().getString(R.string.menu_posterContrast), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionB))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_distorsionB)))
                    mPhotoType.put(getResources().getString(R.string.menu_distorsionB), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_median))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_median)))
                    mPhotoType.put(getResources().getString(R.string.menu_median), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_distorsionC))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_distorsionC)))
                    mPhotoType.put(getResources().getString(R.string.menu_distorsionC), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_sepia))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_sepia)))
                    mPhotoType.put(getResources().getString(R.string.menu_sepia), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_smooth))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_smooth)))
                    mPhotoType.put(getResources().getString(R.string.menu_smooth), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_gaussian))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_gaussian)))
                    mPhotoType.put(getResources().getString(R.string.menu_gaussian), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_bordeado))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_bordeado)))
                    mPhotoType.put(getResources().getString(R.string.menu_bordeado), 1);

            } else if (item.getTitle().equals(getResources().getString(R.string.menu_cartoon))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon)))
                    mPhotoType.put(getResources().getString(R.string.menu_cartoon), 1);
            }
            else if (item.getTitle().equals(getResources().getString(R.string.menu_cartoon_v1))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon_v1)))
                    mPhotoType.put(getResources().getString(R.string.menu_cartoon_v1), 1);
            }
            else if (item.getTitle().equals(getResources().getString(R.string.menu_cartoon_v2))) {
                if (!mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon_v2)))
                    mPhotoType.put(getResources().getString(R.string.menu_cartoon_v2), 1);
            }
            return true;
        }
        if (item.getGroupId() == MENU_GROUP_ID_SEG) {
            if (mImageSizeIndex > 4) {
                Snackbar snackbar = Snackbar.make(findViewById(android.R.id.content),
                        "Can't start. Change size to: " +
                                mSupportedImageSizes.get(mImageSizeIndex).width
                                + " x " +
                                mSupportedImageSizes.get(mImageSizeIndex).height
                                + "or less",
                        Snackbar.LENGTH_LONG);
                snackbar.show();
            }
            mPhotoType.clear();
            if (item.getItemId() == 0)
                mLoadNext = 0;
            if (mPhotoSeg.size() == 2)
                mPhotoSeg.remove(1);
            if (!mPhotoSeg.contains(getResources().getStringArray(R.array.menu_segmentation)[item.getItemId()]))
                mPhotoSeg.add(getResources().getStringArray(R.array.menu_segmentation)[item.getItemId()]);
            mStaticImage = false;
            return true;
        }

        Snackbar snackbar;
        switch (item.getItemId()) {
            case R.id.menu_take_photo:
                //Do not let to use the menu while taking a photo
                mIsMenuLocked = true;
                // FLAG Next frame, take the photo.
                mIsPhotoPending = true;

                return true;
            case R.id.menu_upwards:
                if (!mPhotoType.isEmpty()) {
                    int value = mPhotoType.get(mSelectionValue);
                    mPhotoType.remove(mSelectionValue);
                    value++;
                    mPhotoType.put(mSelectionValue, value);
                    snackbar = Snackbar.make(findViewById(android.R.id.content), "Adjust level " + mSelectionValue + " : " + value, Snackbar.LENGTH_LONG);
                    snackbar.show();
                }
                if (!mPhotoSeg.isEmpty()) {
                    mStaticImage = false;
                    mLoadNext = 1;
                } else {
                    snackbar = Snackbar.make(findViewById(android.R.id.content), "No effects applied", Snackbar.LENGTH_LONG);
                    snackbar.show();
                }
                return true;
            case R.id.menu_downwards:
                if (!mPhotoType.isEmpty()) {
                    int value = mPhotoType.get(mSelectionValue);
                    if (value > 1) {
                        mPhotoType.remove(mSelectionValue);
                        value--;
                        mPhotoType.put(mSelectionValue, value);
                        snackbar = Snackbar.make(findViewById(android.R.id.content), "Adjust level " + mSelectionValue + " : " + value, Snackbar.LENGTH_LONG);
                        snackbar.show();
                    } else {
                        snackbar = Snackbar.make(findViewById(android.R.id.content), "Adjust level " + mPhotoType + " cannot be lower", Snackbar.LENGTH_LONG);
                        snackbar.show();
                    }
                }
                if (!mPhotoSeg.isEmpty()) {
                    mStaticImage = false;
                    mLoadNext = -1;
                } else {
                    snackbar = Snackbar.make(findViewById(android.R.id.content), "No effects applied", Snackbar.LENGTH_LONG);
                    snackbar.show();
                }
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
    public Mat onCameraFrame(final CvCameraViewFrame inputFrame) throws IOException {
        boolean post = true;

        Mat rgba = inputFrame.rgba();

        if (!mStaticImage && !mPhotoSeg.isEmpty()) {
            if (mPhotoSeg.get(0).equals("Load image")) {
                if (mPhotoSeg.size() < 2) {
                    mStaticImage = true;
                    mBgr = rec.loadImage(mLoadNext);
                    Imgproc.resize(mBgr, mBgr,
                            new org.opencv.core.Size(
                                    mSupportedImageSizes.get(mImageSizeIndex).width,
                                    mSupportedImageSizes.get(mImageSizeIndex).height));
                }

                else if (mPhotoSeg.size() == 2 && mPhotoSeg.get(1).equals("Otsu")) {
                    mStaticImage = true;
                    mBgr = rec.otsuThresholding(mBgr, false);
                }

                else if (mPhotoSeg.size() == 2 && mPhotoSeg.get(1).equals("Adaptative")) {
                    mStaticImage = true;
                    mBgr = rec.adaptiveTresholding(mBgr, true);
                }

                else if (mPhotoSeg.size() == 2 && mPhotoSeg.get(1).equals("Contours")) {
                    mStaticImage = true;
                    mBgr = rec.contours(mBgr);
                }
                else if (mPhotoSeg.size() == 2 && mPhotoSeg.get(1).equals("Recognition")) {

                    mStaticImage = true;
                    rec.training("trainingData.txt");
                    resultClas= rec.mahalanobisDistance(mBgr);
                    //mBgr = rec.contours(mBgr);
                    post=false;

                    this.runOnUiThread(new Runnable() {
                        public void run() {
                            AlertDialog alertDialog = new AlertDialog.Builder(MainActivity.this).create();
                            alertDialog.setTitle("Recognition");
                            alertDialog.setIcon(android.R.drawable.ic_dialog_alert);
                            alertDialog.setMessage("0 = Circulo" + resultClas[0] + "\n" +
                                    "1 = Vagon" + resultClas[1] + "\n" +
                                    "2 = Triangulo" + resultClas[2] + "\n" +
                                    "3 = Rectangulo" + resultClas[3] + "\n" +
                                    "4 = Rueda" + resultClas[4] + "\n");
                            alertDialog.setButton(AlertDialog.BUTTON_NEUTRAL, "OK",
                                    new DialogInterface.OnClickListener() {
                                        public void onClick(DialogInterface dialog, int which) {
                                            dialog.dismiss();
                                        }
                                    });
                            alertDialog.show();
                        }
                    });


                }

            } else {
                mPhotoSeg.clear();
                Snackbar snackbar = Snackbar.make(findViewById(android.R.id.content), "Image not loaded", Snackbar.LENGTH_LONG);
                snackbar.show();
            }

        } else {
            if (!mStaticImage) {
                Filters F = new Filters();
                Imgproc.cvtColor(rgba, mBgr, Imgproc.COLOR_RGBA2BGR, 3);

                if (mPhotoType.size() > 0) {
                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_clahe)))
                        mBgr = F.clahe(mBgr, mPhotoType.get(getResources().getString(R.string.menu_clahe)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_heist)))
                        mBgr = F.histEqual(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_alien)))
                        mBgr = F.getSkin(mBgr, mPhotoType.get(getResources().getString(R.string.menu_alien)) % 3);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_alienHSV))) {
                        mBgr = F.alienHSV(mBgr);
                        post = false;
                    }
                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_poster)))
                        mBgr = F.poster(mBgr, mPhotoType.get(getResources().getString(R.string.menu_poster)), 0);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_poster_Ellipse)))
                        mBgr = F.poster(mBgr, mPhotoType.get(getResources().getString(R.string.menu_poster_Ellipse)), 1);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_posterContrast)))
                        mBgr = F.poster2(mBgr, mPhotoType.get(getResources().getString(R.string.menu_posterContrast)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_distorsionB)))
                        mBgr = F.distorsionBarril(mBgr, -mPhotoType.get(getResources().getString(R.string.menu_distorsionB)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_distorsionC)))
                        mBgr = F.distorsionBarril(mBgr, mPhotoType.get(getResources().getString(R.string.menu_distorsionC)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_sepia)))
                        mBgr = F.sepia(mBgr, mPhotoType.get(getResources().getString(R.string.menu_sepia)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_smooth)))
                        mBgr = F.boxBlur(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_median)))
                        mBgr = F.medianBlur(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_gaussian)))
                        mBgr = F.gaussianSmooth(mBgr, mPhotoType.get(getResources().getString(R.string.menu_gaussian)));

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_bordeado)))
                        mBgr = F.bordeado(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon)))
                        mBgr = F.cartoon(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon_v1)))
                        mBgr = F.cart_p1(mBgr);

                    if (mPhotoType.containsKey(getResources().getString(R.string.menu_cartoon_v2)))
                        mBgr = F.cart_p2(mBgr);

                    if (post) {
                        Imgproc.cvtColor(mBgr, rgba, Imgproc.COLOR_BGR2RGBA, 3);
                    } else {
                        return mBgr;
                    }
                }
            }
        }
        if (mIsPhotoPending) {
            mIsPhotoPending = false;
            takePhoto();
        }

            if (post) {
                Imgproc.cvtColor(mBgr, rgba, Imgproc.COLOR_BGR2RGBA);
                return rgba;
            } else
                return mBgr;


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
