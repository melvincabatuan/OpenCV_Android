package vision.computer.opencv_android.training;

import java.io.Serializable;
import java.util.ArrayList;

/**
 * Created by Daniel on 20/03/2016.
 */
public class DataSender implements Serializable {

    private ArrayList<double[]> values = new ArrayList<>();

    public DataSender(ArrayList<double[]> data) {
        this.values = data;
    }

    public ArrayList<double[]> getParliaments() {
        return this.values;
    }

}