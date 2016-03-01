package vision.computer.opencv_android;

import org.opencv.core.Mat;

/**
 * Created by Manuel on 29/02/2016.
 */
public class ConvolutionMatrix {
    public static int SIZE = 3;

    public double[][] Matrix;
    public double factor = 1;
    public double offset = 1;
    private int value = 0;

    public ConvolutionMatrix(int size) {
        this.SIZE = size;
        Matrix = new double[SIZE][SIZE];
    }

    public static Mat computeConvolution3x3(Mat src, ConvolutionMatrix matrix) {
        int rows = src.rows();
        int cols = src.cols();

        double A, R, G, B;
        double sumR, sumG, sumB;

        //rgba{pixel I, pixel J, Color BGR}
        double[][][] rgba = new double[SIZE][SIZE][src.channels()];

        for (int y = 0; y < cols - 2; ++y) {
            for (int x = 0; x < rows - 2; ++x) {
                // Get pixels on the matrix.
                for (int i = 0; i < SIZE; ++i)
                    for (int j = 0; j < SIZE; ++j)
                        rgba[i][j] = src.get(x + i, y + j);

                // init color sum
                sumR = sumG = sumB = 0;

                // get sum of RGB on matrix
                for (int i = 0; i < SIZE; ++i) {
                    for (int j = 0; j < SIZE; ++j) {
                        sumR += (rgba[i][j][2] * matrix.Matrix[i][j]);
                        sumG += (rgba[i][j][1] * matrix.Matrix[i][j]);
                        sumB += (rgba[i][j][0] * matrix.Matrix[i][j]);
                    }
                }

                // get final Red
                R = sumR / matrix.value;
                if (R < 0)
                    R = 0;
                else if (R > 255)
                    R = 255;

                // get final Green
                G = sumG / matrix.value;
                if (G < 0)
                    G = 0;
                else if (G > 255)
                    G = 255;

                // get final Blue
                B = sumB / matrix.value;
                if (B < 0)
                    B = 0;
                else if (B > 255)
                    B = 255;

                src.put(x + 1, y + 1, new double[]{B, G, R});
            }
        }
        // final image
        return src;
    }

    public void setAll(double value) {
        for (int x = 0; x < SIZE; ++x) {
            for (int y = 0; y < SIZE; ++y) {
                Matrix[x][y] = value;
                this.value += value;
            }
        }
    }

    public void applyConfig(double[][] config) {
        for (int x = 0; x < SIZE; ++x) {
            for (int y = 0; y < SIZE; ++y) {
                Matrix[x][y] = config[x][y];
                this.value += config[x][y];
            }
        }
    }
}