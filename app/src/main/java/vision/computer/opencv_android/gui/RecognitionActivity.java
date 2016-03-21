package vision.computer.opencv_android.gui;

import android.content.Intent;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.widget.TextView;

import java.util.ArrayList;

import vision.computer.opencv_android.R;
import vision.computer.opencv_android.training.DataSender;

public class RecognitionActivity extends AppCompatActivity {

    public static final String EXTRA_RECIEVE_CIRCULO = "recognitioncircle";
    public static final String EXTRA_RECIEVE_RECTANGULO = "recognitionrectangulo";
    public static final String EXTRA_RECIEVE_VAGON = "recognitionvagon";
    public static final String EXTRA_RECIEVE_RUEDA = "recognitionrueda";
    public static final String EXTRA_RECIEVE_TRIANGULO = "recognitiontriangulo";
    public static final String EXTRA_RECIEVE_NUM = "numobjects";
    private final double THRESHOLD = 12.8; //CHI-SQUARE --> m=5 alpha=0.025
    private ArrayList<double[]> resultCirculo = new ArrayList<>();
    private ArrayList<double[]> resultTriangulo = new ArrayList<>();
    private ArrayList<double[]> resultRectangulo = new ArrayList<>();
    private ArrayList<double[]> resultRueda = new ArrayList<>();
    private ArrayList<double[]> resultVagon = new ArrayList<>();
    private ArrayList<ArrayList<double[]>> results = new ArrayList<>();
    private int numObjects;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        final Intent intent = getIntent();

        String result = "===== RESULT =====\n\n";

        numObjects = intent.getIntExtra(EXTRA_RECIEVE_NUM, 0);
        DataSender dat = (DataSender) intent.getSerializableExtra(EXTRA_RECIEVE_CIRCULO);
        results.add(dat.getParliaments());
        dat = (DataSender) intent.getSerializableExtra(EXTRA_RECIEVE_RECTANGULO);
        results.add(dat.getParliaments());
        dat = (DataSender) intent.getSerializableExtra(EXTRA_RECIEVE_RUEDA);
        results.add(dat.getParliaments());
        dat = (DataSender) intent.getSerializableExtra(EXTRA_RECIEVE_TRIANGULO);
        results.add(dat.getParliaments());
        dat = (DataSender) intent.getSerializableExtra(EXTRA_RECIEVE_VAGON);
        results.add(dat.getParliaments());

        setContentView(R.layout.activity_recognition);

        TextView tv = (TextView) findViewById(R.id.text);
        TextView res = (TextView) findViewById(R.id.result);
        String write = "";
        String[] objects = new String[]{"circulo ", "vagon ", "triangulo ", "rectangulo ", "rueda "};
        String[] parameters = new String[]{"area ", "perimeter ", "huM-1 ", "huM-2 ", "huM-3 "};

        for (int x = 0; x < numObjects; x++) {
            write += "===== Object: " + x + " ===== \n\n";
            for (int i = 0; i < objects.length; i++) {
                write += objects[i];
                boolean recognize = true;
                for (int j = 0; j < parameters.length; j++) {
                    write += parameters[j];
                    write += results.get(i).get(x)[j] + " ";
                    if (results.get(i).get(x)[j] > THRESHOLD){
                        recognize = false;
                        break;
                    }

                }
                if (recognize)
                    result += "Object " + x + " -->" + objects[i];
                write += "\n\n";
            }
        }
        tv.setText(write);
        res.setText(result);
    }

    public boolean recognizeClass(double[] values) {
        boolean recog = true;
        for (int i = 0; i < values.length; i++) {
            if (values[i] > THRESHOLD) {
                recog = false;
            }
        }
        return recog;
    }
}
