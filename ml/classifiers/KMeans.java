package ml.classifiers;

import java.util.ArrayList;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

public class KMeans implements Classifier{
    int k;
    int iterations;
    ArrayList<Example> centroids;
    DataSet data;

    public KMeans(int k){
        this.k = k;
        this.iterations = 50;
        centroids = new ArrayList<>();
    }

    @Override
    public void train(DataSet data) {
        // initalize centers randomly
        this.data = data;
        this.centroids = this.initalizeCentroids();

        for(int i=0;i<iterations;i++){
            // assign points to nearest center

            // recalculate centers
        }
    }

    private ArrayList<Example> initalizeCentroids(){
        Random rand = new Random();
        ArrayList<Example> curCentroids = new ArrayList<>();

        for(int i=0;i<k;i++){
            Example newCentroid = new Example();
            for(Integer feature : data.getAllFeatureIndices()){
                int featureValue = rand.nextInt(1);
                if(featureValue > 0){
                    newCentroid.addFeature(feature, featureValue);
                }        
            }

            curCentroids.add(newCentroid);
        }

        return curCentroids;
    }

    private void recalculateCentroid(){
        
    }

    private double eucDist(Example e1, Example e2){
        double sum = 0;

        for (Integer feature : e1.getFeatureSet()) {
            if(e1.getFeatureSet().contains(feature) && e2.getFeatureSet().contains(feature)){
                Double v1 = e1.getFeature(feature);
                Double v2 = e2.getFeature(feature);
                sum += Math.pow(v1 - v2, 2);
            }
        }

        return Math.sqrt(sum);
    }

    private double cosDist(Example e1, Example e2){
        double numerator = 0;
        double left = 0;
        double right = 0;
        double denominator = 0;

        for (Integer feature : e1.getFeatureSet()) {
            if(e1.getFeatureSet().contains(feature) && e2.getFeatureSet().contains(feature)){
                Double v1 = e1.getFeature(feature);
                Double v2 = e2.getFeature(feature);
                numerator+=v1*v2;
            }
        }

        for (Integer feature : e1.getFeatureSet()) {
            left+=Math.pow(e1.getFeature(feature),2);
        }

        for (Integer feature : e2.getFeatureSet()) {
            right+=Math.pow(e2.getFeature(feature),2);
        }

        denominator = Math.sqrt(left)*Math.sqrt(right);
        return 1-numerator/denominator;
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
