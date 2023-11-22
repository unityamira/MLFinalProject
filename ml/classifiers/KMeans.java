package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

public class KMeans{
    int k;
    int iterations;
    DataSet data;

    ArrayList<Example> centroids;
    HashMap<Integer, ArrayList> associatedPoints;

    private Integer COSINE = 0;
    private Integer EUCLIDEAN = 0;
    private int distChoice;

    public KMeans(int k){
        this.k = k;
        this.iterations = 50;
        this.distChoice = 0;
        this.centroids = new ArrayList<>();
    }

    public void train(DataSet data) {
        // initalize centers randomly
        this.data = data;
        this.centroids = this.initalizeCentroids();

        for(int iteration=0;iteration<iterations;iteration++){
            ArrayList<Example> examples = data.getData();

            // assign points to nearest center
            for(int i=0;i<examples.size();i++){
                int nearestCent = (int) nearestCentroid(examples.get(i));
                ArrayList<Example> curPoints = associatedPoints.get(nearestCent);
                curPoints.add(examples.get(i));
                associatedPoints.put(nearestCent, curPoints);
            }

            // recalculate centers
            for(int i=0;i<centroids.size();i++){
                this.recalculateCentroid(centroids.get(i));
            }
        }
    }

    /**
     * 
     * @param e
     * @return
     */
    private double nearestCentroid(Example e){
        double minimumDistance = Double.MAX_VALUE;
        Example nearest = null;

        for (Example centroid : centroids) {
            double currentDistance = 0;

            if(distChoice == COSINE){
                currentDistance = cosDist(e, centroid);
            }else if(distChoice == EUCLIDEAN){
                currentDistance = eucDist(e, centroid);
            }
            

            if (currentDistance < minimumDistance) {
                minimumDistance = currentDistance;
                nearest = centroid;
            }
        }

        return nearest.getLabel();
    }

    /**
     * 
     * @return
     */
    public ArrayList<Example> initalizeCentroids(){
        Random rand = new Random();
        ArrayList<Example> curCentroids = new ArrayList<>();

        for(int i=0;i<k;i++){
            Example newCentroid = new Example();
            ArrayList<Example> points = new ArrayList<>();
            associatedPoints.put(i, points);
            newCentroid.setLabel(i);

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

    /**
     * 
     * @param e
     */
    private void recalculateCentroid(Example e){
        ArrayList<Example> points = associatedPoints.get(e.getLabel());
        HashMap<Integer, Double> averages = new HashMap<>();
        
        for(int i=0;i<points.size();i++){
            Example curPoint = points.get(i);
            for(Integer feature : curPoint.getFeatureSet()){
                if(averages.containsKey(feature)){
                    averages.put(feature, averages.get(feature)+curPoint.getFeature(feature));
                }else{
                    averages.put(feature, curPoint.getFeature(feature));
                }
            }
        }

        for(Integer feature : data.getAllFeatureIndices()){
            if(averages.containsKey(feature)){
                averages.put(feature, averages.get(feature)/points.size());
            }
        }
    }

    /**
     * Choose type of distance used to calculate closes point
     * @param distChoice
     */
    public void chooseDistance(int distChoice){
        this.distChoice = distChoice;
    }

    /**
     * 
     * @param e1
     * @param e2
     * @return
     */
    public double eucDist(Example e1, Example e2){
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

    /**
     * 
     * @param e1
     * @param e2
     * @return
     */
    public double cosDist(Example e1, Example e2){
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

    public static void main(String[] args){
        Random rand = new Random();

        KMeans model = new KMeans(3);
        DataSet wineDataSet = new DataSet("data/wines.train", DataSet.TEXTFILE);
        DataSet simpleDataSet = new DataSet("data/simple.csv", DataSet.CSVFILE);
        ArrayList<Example> examples = wineDataSet.getData();

        Example e1 = simpleDataSet.getData().get(0);
        Example e2 = simpleDataSet.getData().get(1);
        Example e3 = simpleDataSet.getData().get(2);
        System.out.print("E1: ");
        System.out.println(e1);
        System.out.print("E2: ");
        System.out.println(e2);
        System.out.print("Cosine Distance:");
        System.out.println(model.cosDist(e1, e2));
        System.out.print("Euclidean Distance:");
        System.out.println(model.eucDist(e1, e2));
        System.out.println("");

        System.out.print("E1: ");
        System.out.println(e1);
        System.out.print("E2: ");
        System.out.println(e3);
        System.out.print("Cosine Distance:");
        System.out.println(model.cosDist(e1, e3));
        System.out.print("Euclidean Distance:");
        System.out.println(model.eucDist(e1, e3));


        for(int i=0;i<1;i++){
            Example ex1 = examples.get(rand.nextInt(wineDataSet.getData().size()));
            Example ex2 = examples.get(rand.nextInt(wineDataSet.getData().size()));

            System.out.println(ex1.toString(wineDataSet.getFeatureMap())+"\n");
            System.out.println(ex2.toString(wineDataSet.getFeatureMap())+"\n");

            System.out.print("Cosine Distance:");
            System.out.println(model.cosDist(ex1, ex2));
            System.out.print("Euclidean Distance:");
            System.out.println(model.eucDist(ex1, ex2));
        }
    }
}