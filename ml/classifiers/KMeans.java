package ml.classifiers;

import ml.data.DataSet;
import ml.data.Example;

public class KMeans implements Classifier{
    int k;
    int iterations;

    public KMeans(int k){
        this.k = k;
        this.iterations = 50;
    }

    @Override
    public void train(DataSet data) {
        // initalize centers randomly

        for(int i=0;i<iterations;i++){
            // assign points to nearest center

            // recalculate centers
        }
    }

    @Override
    public double classify(Example example) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'classify'");
    }

    @Override
    public double confidence(Example example) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'confidence'");
    }
}
