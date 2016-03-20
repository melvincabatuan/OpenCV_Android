package vision.computer.opencv_android.gui;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.widget.TextView;

import vision.computer.opencv_android.R;

public class RecognitionActivity extends AppCompatActivity {

    public static final String EXTRA_RECIEVE_CIRCULO = "recognitioncircle";
    public static final String EXTRA_RECIEVE_RECTANGULO = "recognitionrectangulo";
    public static final String EXTRA_RECIEVE_VAGON = "recognitionvagon";
    public static final String EXTRA_RECIEVE_RUEDA = "recognitionrueda";
    public static final String EXTRA_RECIEVE_TRIANGULO = "recognitiontriangulo";

    private double[][] mRecognition = new double[5][5];

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final Intent intent = getIntent();

        mRecognition[0] = intent.getDoubleArrayExtra(EXTRA_RECIEVE_CIRCULO);
        mRecognition[1] = intent.getDoubleArrayExtra(EXTRA_RECIEVE_RECTANGULO);
        mRecognition[2] = intent.getDoubleArrayExtra(EXTRA_RECIEVE_RUEDA);
        mRecognition[3] = intent.getDoubleArrayExtra(EXTRA_RECIEVE_TRIANGULO);
        mRecognition[4] = intent.getDoubleArrayExtra(EXTRA_RECIEVE_VAGON);


        setContentView(R.layout.activity_recognition);
        Toolbar toolbar = (Toolbar) findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        TextView tv = (TextView) findViewById(R.id.text);
        String write = "";
        String[] objects = new String[]{"circulo ", "vagon ", "triangulo ", "rectangulo ", "rueda "};
        String[] parameters = new String[]{"area ", "perimeter ", "huM-1 ", "huM-2 ", "huM-3 "};

        for (int i = 0; i < mRecognition.length; i++) {
            write += objects[i];
            for (int j = 0; j < mRecognition[i].length; j++) {
                write += parameters[j];
                write += mRecognition[i][j] + " ";
            }
            write += "\n\n";
        }

        tv.setText(write);
    }

}
