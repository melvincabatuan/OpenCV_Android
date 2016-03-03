package vision.computer.opencv_android;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.CLAHE;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.util.ArrayList;
import java.util.List;

/**
 * Created by Manuel on 28/02/2016.
 */
public class Filters {

    final static double e = 2.71828;

    public Filters() {
    }

    public static Mat gaussianSmooth(Mat src, int ro) {
        /*Mat MorphKernel = new Mat();
        double pi = 3.1416;
        double coef = (1/(2*pi*ro*ro));
        double[][] res = new double[size][size];

        for (int i = -offset; i < size - offset; i++) {
            for (int j = -offset; j < size - offset; j++) {
                //res[i + offset][j + offset] = Math.pow(e, -((i * i + j * j) / (double)(2 * ro * ro)));
                //coef += res[i + offset][j + offset];
                MorphKernel.put(i, j, Math.pow(e, -((i * i + j * j) / (double)(2 * ro * ro))) * coef);
            }
        }
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++)
                MorphKernel.put(i, j, res[i][j] * (1/coef));
        }*/
        //Imgproc.morphologyEx(src, src, Imgproc.MORPH_RECT, MorphKernel);
        Size size = new Size(5 * ro - 7 * ro, 5 * ro - 7 * ro);
        Imgproc.GaussianBlur(src, src, size, ro, ro);

        return src;
    }

    public static Mat medianBlur(Mat src) {
        int size = 5;
        Imgproc.medianBlur(src, src, size);
        return src;
    }

    public static Mat boxBlur(Mat src) {
       /* Mat MorphKernel = new Mat();
        int size = 5;
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                MorphKernel.put(i, j, 1);
            }
        }

        //Applies the morfological operation MORPH_RECT to a MAT given a Kernel
        Imgproc.morphologyEx(src, src, Imgproc.MORPH_RECT, MorphKernel);*/
        Imgproc.boxFilter(src, src, -1, new Size(3, 3));
        return src;

    }

    public Mat sepia(Mat bgr, int depth) {
        // Sepia values
        double red = 60;
        double green = 35;
        double blue = 0;

        int rows = bgr.rows();
        int cols = bgr.cols();

        // constant grayscale
        final double GS_RED = 0.3;
        final double GS_GREEN = 0.59;
        final double GS_BLUE = 0.11;

        double[] pixel = new double[3];

        // scan through all pixels
        for (int x = 0; x < rows; ++x) {
            for (int y = 0; y < cols; ++y) {

                // get pixel color
                pixel = bgr.get(x, y);
                // apply grayscale sample
                pixel[0] = pixel[1] = pixel[2] = (int) (GS_RED * pixel[2] + GS_GREEN * pixel[1] + GS_BLUE * pixel[0]);
                // apply intensity level for sepid-toning on each channel
                pixel[2] += (depth * red);
                if (pixel[2] > 255) {
                    pixel[2] = 255;
                }
                pixel[1] += (depth * green);
                if (pixel[1] > 255) {
                    pixel[1] = 255;
                }
                pixel[0] += (depth * blue);
                if (pixel[0] > 255) {
                    pixel[0] = 255;
                }
                // set new pixel color to output image
                bgr.put(x, y, pixel);
            }
        }
        // return final image
        return bgr;
    }

    public Mat alienHSV(Mat bgr) {
        Mat hsv = new Mat();
        Mat res = new Mat();
        double scaleSatLower = 0.28;
        double scaleSatUpper = 0.68;

        Imgproc.cvtColor(bgr, hsv, Imgproc.COLOR_BGR2HSV);
        Scalar lower = new Scalar(0, scaleSatLower * 255, 0);
        Scalar upper = new Scalar(25, scaleSatUpper * 255, 255);
        Core.inRange(hsv, lower, upper, res);
        return res;
    }

    public Mat facialDetection(Mat bgr, Mat grayscaleImage, CascadeClassifier cascadeClassifier, int absoluteFaceSize) {
        Imgproc.cvtColor(bgr, grayscaleImage, Imgproc.COLOR_RGBA2RGB);
        MatOfRect faces = new MatOfRect();


        // Use the classifier to detect faces
        if (cascadeClassifier != null) {
            cascadeClassifier.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2, new org.opencv.core.Size(absoluteFaceSize, absoluteFaceSize), new org.opencv.core.Size());
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++)
            Imgproc.rectangle(bgr, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0, 255), 3);
        return bgr;
    }

    private boolean R1(double R, double G, double B) {
        boolean e1 = (R > 95) && (G > 40) && (B > 20) && ((Math.max(R, Math.max(G, B)) - Math.min(R, Math.min(G, B))) > 15) && (Math.abs(R - G) > 15) && (R > G) && (R > B);
        boolean e2 = (R > 220) && (G > 210) && (B > 170) && (Math.abs(R - G) <= 15) && (R > B) && (G > B);
        return (e1 || e2);
    }

    private boolean R2(float Y, float Cr, float Cb) {
        boolean e3 = Cr <= 1.5862 * Cb + 20;
        boolean e4 = Cr >= 0.3448 * Cb + 76.2069;
        boolean e5 = Cr >= -4.5652 * Cb + 234.5652;
        boolean e6 = Cr <= -1.15 * Cb + 301.75;
        boolean e7 = Cr <= -2.2857 * Cb + 432.85;
        return e3 && e4 && e5 && e6 && e7;
    }

    private boolean R3(float H, float S, float V) {
        return (H < 25) || (H > 230);
    }

    public Mat getSkin(Mat src, int color) {
        // allocate the result matrix
        Mat dst = src.clone();
        byte[] cblack = new byte[src.channels()];
        for (int i = 0; i < src.channels(); i++) {
            cblack[i] = Byte.MIN_VALUE;
        }
        byte[] cred = new byte[3];
        cred[0] = 0;
        cred[1] = 0;
        cred[2] = Byte.MAX_VALUE;


        Mat src_ycrcb = new Mat(), src_hsv = new Mat();
        // OpenCV scales the YCrCb components, so that they
        // cover the whole value range of [0,255], so there's
        // no need to scale the values:
        Imgproc.cvtColor(src, src_ycrcb, Imgproc.COLOR_BGR2YCrCb);
        // OpenCV scales the Hue Channel to [0,180] for
        // 8bit images, so make sure we are operating on
        // the full spectrum from [0,360] by using floating
        // point precision:
        src.convertTo(src_hsv, CvType.CV_32FC3);
        Imgproc.cvtColor(src_hsv, src_hsv, Imgproc.COLOR_BGR2HSV);
        // Now scale the values between [0,255]:
        Core.normalize(src_hsv, src_hsv, 0.0, 255.0, Core.NORM_MINMAX, CvType.CV_32FC3);

        for (int i = 0; i < src.rows(); i++) {
            for (int j = 0; j < src.cols(); j++) {
                double[] pix_bgr = new double[3];
                pix_bgr = src.get(i, j);
                double B = pix_bgr[0];
                double G = pix_bgr[1];
                double R = pix_bgr[2];

                // apply rgb rules
                boolean a = R1(R, G, B);

                double[] pix_ycrcb = new double[3];
                pix_ycrcb = src_ycrcb.get(i, j);
                double Y = pix_ycrcb[0];
                double Cr = pix_ycrcb[1];
                double Cb = pix_ycrcb[2];
                // apply ycrcb rule
                boolean b = R2((float) Y, (float) Cr, (float) Cb);

                double[] pix_hsv = new double[3];
                pix_hsv = src_hsv.get(i, j);
                float H = (float) pix_hsv[0];
                float S = (float) pix_hsv[1];
                float V = (float) pix_hsv[2];
                // apply hsv rule
                boolean c = R3(H, S, V);

                if ((a && b && c)) {
                    double[] pixeldst = new double[3];
                    pixeldst[0] = B;
                    pixeldst[1] = G;
                    pixeldst[2] = R;
                    pixeldst[color] += 100;
                    if (pixeldst[color] > 255)
                        pixeldst[color] = 255;
                    dst.put(i, j, pixeldst);
                } else {
                }
            }
        }
        return dst;
    }

    private double pixelMaximization(double pixel, int max, int min) {
        if (pixel > max / 2)
            pixel = max;
        else
            pixel = min;
        return pixel;
    }

    public Mat poster2(Mat bgr, int index) {
        int MAXCOLOR = 255;
        int colorCoef = (MAXCOLOR / index)+1;
        ArrayList<Integer> splitted = new ArrayList<Integer>();
        ;
        for (int w = 0; w <= index; w++) {
            int value = (MAXCOLOR / index * w);
            splitted.add(value);
        }
        for (int i = 0; i < bgr.rows(); i++) {
            for (int j = 0; j < bgr.cols(); j++) {
                double[] pixel = bgr.get(i, j);
                for (int c = 0; c < bgr.channels(); c++) {
                    pixel[c] = pixelMaximization(pixel[c], splitted.get((int) (pixel[c] / colorCoef + 1)),
                            splitted.get((int) (pixel[c] / colorCoef)));
                }
                bgr.put(i, j, pixel);
            }
        }
        return bgr;
    }

    public Mat poster(Mat bgr, int size, int m) {
        int morph = -1;
        if (m == 0)
            morph = Imgproc.MORPH_CROSS;
        if (m == 1)
            morph = Imgproc.MORPH_ELLIPSE;

        org.opencv.core.Size ksize = new org.opencv.core.Size(size, size);

        //Gives a CROSS-KERNEL with the size specified.
        Mat MorphKernel = Imgproc.getStructuringElement(morph, ksize);

        //Applies the morfological operation MORPH_RECT to a MAT given a Kernel
        Imgproc.morphologyEx(bgr, bgr, Imgproc.MORPH_RECT, MorphKernel);
        return bgr;
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
    public Mat clahe(Mat bgr, int limit) {
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
    public Mat histEqual(Mat bgr) {
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

    public Mat distorsionBarril(Mat bgr, int k) {
        Mat map_x = new Mat(), map_y = new Mat(), output = new Mat();
        double Cy = (double) bgr.cols() / 2;
        double Cx = (double) bgr.rows() / 2;
        map_x.create(bgr.size(), CvType.CV_32FC1);
        map_y.create(bgr.size(), CvType.CV_32FC1);

        for (int x = 0; x < map_x.rows(); x++) {
            for (int y = 0; y < map_y.cols(); y++) {
                double r2 = (x - Cx) * (x - Cx) + (y - Cy) * (y - Cy);
                double data = ((y - Cy) / (1 + (k / 1000000.0) * r2) + Cy); // se suma para obtener la posicion absoluta
                map_x.put(x, y, data);
                double data2 = ((x - Cx) / (1 + (k / 1000000.0) * r2) + Cx); // la posicion relativa del punto al centro
                map_y.put(x, y, data2);
            }
        }
        Imgproc.remap(bgr, output, map_x, map_y, Imgproc.INTER_LINEAR);
        return output;
    }

    public Mat bordeado(Mat bgr) {
        Mat dst = new Mat();
        Mat inv = new Mat();
        Imgproc.cvtColor(bgr, bgr, Imgproc.COLOR_BGR2GRAY);
        Imgproc.Sobel(bgr, inv, CvType.CV_32F, 1, 0);

        Core.MinMaxLocResult res = Core.minMaxLoc(inv);
        inv.convertTo(dst, CvType.CV_8U, 255.0 / (res.maxVal - res.minVal), -res.minVal * 255.0 / (res.maxVal - res.minVal));
        Core.bitwise_not(dst, dst);
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_GRAY2BGR);
        return dst;
    }

    public Mat cartoon(Mat bgr) {
        int height = bgr.height();
        int width = bgr.width();
        Mat image = new Mat();
        Imgproc.cvtColor(bgr, bgr, Imgproc.COLOR_BGR2RGB);
        for (int i = 0; i < 2; i++) {
            Imgproc.pyrDown(bgr, image);
        }
        bgr.release();
        Mat image_bi = new Mat();
        for (int j = 0; j < 7; j++) {
            Imgproc.bilateralFilter(image, image_bi, 9, 9, 7);
        }
        image.release();
        for (int i = 0; i < 2; i++) {
            Imgproc.pyrUp(image_bi, image_bi);
        }
        Mat gray = new Mat();
        Mat blur = new Mat();
        Imgproc.cvtColor(image_bi, gray, Imgproc.COLOR_RGB2GRAY);
        Imgproc.medianBlur(gray, blur, 7);
        gray.release();
        Mat edge = new Mat();
        Imgproc.adaptiveThreshold(blur, edge, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY, 9, 2);
        blur.release();
        Imgproc.cvtColor(edge, edge, Imgproc.COLOR_GRAY2RGB);
        Mat cartoon = new Mat();
        Core.bitwise_and(image_bi, edge, cartoon);
        //Imgproc.cvtColor(cartoon, cartoon, Imgproc.COLOR_RGB2BGR);
        Imgproc.cvtColor(cartoon, cartoon, Imgproc.COLOR_RGB2BGR);
        Imgproc.resize(cartoon, cartoon, new Size(width, height));
        return cartoon;
    }

}

