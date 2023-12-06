package ml.classifiers;

import java.util.*;
import ml.data.*;

public class KMeans implements Classifier{
    private int k;
    private int iterations;
    private int distChoice;
    private int initChoice;
    private DataSet data;
    private Random rand;

    ArrayList<Centroid> centroids;
    ArrayList<Example> examples;
    HashMap<Integer, Double> featureMins;
    HashMap<Integer, Double> featureMaxes;

    public static final int COSINE_DIST = 0;
    public static final int EUCLIDEAN_DIST = 1;

    public static final int RANDOM_INIT = 0;
    public static final int FARTHEST_CENT_INIT = 1;

    public static final int ENTROPY = 0;
    public static final int PURITY = 1;
    public static final int SSE = 2;
    public static final int SILHOUETTE = 3;

    public KMeans(int k){
        this.rand = new Random();
        this.k = k;
        this.iterations = 1;
        this.distChoice = EUCLIDEAN_DIST;
        this.initChoice = RANDOM_INIT;
        this.centroids = new ArrayList<>();
        this.featureMins = new HashMap<>();
        this.featureMaxes = new HashMap<>();
    }

    public void train(DataSet data) {
        // initalize centers randomly
        this.data = data;
        this.examples = data.getData();

        if(initChoice == RANDOM_INIT){
            this.initalizeRandom();
        }else if(initChoice == FARTHEST_CENT_INIT){
            this.initalizeFarthest();
        }
        
        for(int iteration=0;iteration<iterations;iteration++){
            // wipe the slate clean
            Collections.shuffle(examples);
            for(int i=0;i<centroids.size();i++){
                centroids.get(i).clearExamples();
            }            

            // assign points to nearest center
            for(int i=0;i<examples.size();i++){
                Example curExample = examples.get(i);
                this.nearestCentroid(curExample).addExample(curExample);
            }

            // for each centroid, recalculate
            for(int i=0;i<centroids.size();i++){
                this.recalculateCentroid(centroids.get(i));
            }

            /*
            System.out.println("Iteration");
            for(int i=0;i<centroids.size();i++){
                System.out.println(centroids.get(i));
                System.out.println(centroids.get(i).getAssociatedPoints().size());
            }
            */

            //System.out.println(averageScore(SSE));
        }
    }

    /**
     * For all centroids, find the one closest to the given datapoint
     * @param e a particular datapoint
     * @return nearest Centroid
     */
    private Centroid nearestCentroid(Example e){
        double minimumDistance = Double.MAX_VALUE;
        Centroid nearest = null;

        for (Centroid curCentroid : centroids) {
            double currentDistance = 0;

            if(distChoice == COSINE_DIST){
                currentDistance = cosineDistance(e, curCentroid);
            }else if(distChoice == EUCLIDEAN_DIST){
                currentDistance = euclideanDist(e, curCentroid);
            }

            if (currentDistance < minimumDistance) {
                minimumDistance = currentDistance;
                nearest = curCentroid;
            }
        }

        return nearest;
    }

    /**
     * For a data point, return the 2nd nearest centroid
     * Used in silhouette analysis
     * @param e example you're interested in
     * @return 2nd nearest centroid
     */
    private Centroid nextNearestCentroid(Example e){
        Centroid first = new Centroid();
        Centroid second = new Centroid();

        double min = Double.MAX_VALUE;
        double secMin = Double.MAX_VALUE;

        for (Centroid curCentroid : centroids) {
            double curDist = 0;
            if(distChoice == COSINE_DIST){
                curDist = cosineDistance(e, curCentroid);
            }else if(distChoice == EUCLIDEAN_DIST){
                curDist = euclideanDist(e, curCentroid);
            }

            if(curDist < min){
                secMin = min;
                second = first;

                first = curCentroid;
                min = curDist;
            }else if(curDist < secMin){
                second = curCentroid;
                secMin = curDist;
            }
        }
        
        return second;
    }

    /**
     * Instantiates k random centroids
     */
    public void initalizeRandom(){
        this.findFeatureRange();
        HashSet<Integer> indices = new HashSet<>();

        for(int i=0;i<k;i++){
            int newIdx = rand.nextInt(examples.size());
            if(indices.contains(newIdx)){
                k-=1;
            }else{
                Centroid curCentroid = new Centroid(examples.get(newIdx));
                curCentroid.setLabel(i);
                indices.add(newIdx);
                centroids.add(curCentroid);
            }

            /*
            // for each feature, randomly assign within training range
            for(Integer feature : data.getAllFeatureIndices()){
                double featureValue = Math.random() * (featureMaxes.get(feature)-featureMins.get(feature)) + featureMins.get(feature);
                if(featureValue != 0){
                    curCentroid.addFeature(feature, featureValue);
                }  
            }
            */

            
        }
    }

    /**
     * Initalizes k centroids using farthest centers heuristic
     * Also know as Kmeans++
     */
    private void initalizeFarthest(){
        // choose random point 
        Example firstCenter = examples.get(this.rand.nextInt(examples.size()));
        Centroid curCentroid = new Centroid(firstCenter);
        curCentroid.setLabel(0);
        centroids.add(curCentroid);

        // for k-1, choose the example that is farthest from already chosen centroids
        for(int i = 0; i < k-1 ;i++){
            
            Centroid nextCentroid = new Centroid(examples.get(findFarthestPoint()));
            nextCentroid.setLabel(i+1);
            //System.out.println(nextCentroid);
            centroids.add(nextCentroid);
        }
    }

    /**
     * For currently set centroids, find the point farthest from all centroids
     */
    private int findFarthestPoint(){
        int maxIndex = 0; double curMax = 0;

        for(int j=0;j<examples.size();j++){
            Example curExample = examples.get(j);
            double d = Double.MAX_VALUE;

            for(int cid = 0;cid < centroids.size();cid++){
                if(distChoice == EUCLIDEAN_DIST){
                    d = Math.min(d, euclideanDist(centroids.get(cid), curExample));
                }else if(distChoice == COSINE_DIST){
                    d = Math.min(d, cosineDistance(centroids.get(cid), curExample));
                }
            }

            if(d > curMax){
                curMax = d;
                maxIndex = j;
            }
        }   
        
        return maxIndex;
    }

    /**
     * For each feature, finds minimum and maximum value within dataset
     */
    private void findFeatureRange(){
        // for all examples and for all feature values, find the min and max value
        for(int i=0;i<examples.size();i++){
            Example curExample = examples.get(i);
            
            for(Integer curFeature : curExample.getFeatureSet()){
                if(featureMins.containsKey(curFeature)){
                    featureMins.put(curFeature, Math.min(curExample.getFeature(curFeature), featureMins.get(curFeature)));
                }else{
                    featureMins.put(curFeature, curExample.getFeature(curFeature));
                }

                if(featureMaxes.containsKey(curFeature)){
                    featureMaxes.put(curFeature, Math.max(curExample.getFeature(curFeature), featureMaxes.get(curFeature)));
                }else{
                    featureMaxes.put(curFeature, curExample.getFeature(curFeature));
                }
            }
        }
    }

    /**
     * Recalculates the centroid given all its associated points
     * @param curCentroid centroid to be recalculated
     */
    private void recalculateCentroid(Centroid curCentroid){
        ArrayList<Example> points = curCentroid.getAssociatedPoints();
        HashMap<Integer, Double> averages = new HashMap<>();
        
        // for all points, sum each feature value
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

        // for all features, divide by the number of points
        // set each centroid feature to be the average
        for(Integer feature : data.getAllFeatureIndices()){
            if(averages.containsKey(feature)){
                averages.put(feature, averages.get(feature)/points.size());
                curCentroid.setFeature(feature, averages.get(feature));
            }
        }
    }

    /**
     * Choose type of distance used to calculate closest point
     * @param distChoice
     */
    public void chooseDistance(int distChoice){
        this.distChoice = distChoice;
    }

    /**
     * Choose type of initalization for centroids
     * @param initChoice choice of initalization
     */
    public void chooseInitialize(int initChoice){
        this.initChoice = initChoice;
    }

    /**
     * Choose number of iterations
     * @param iterations 
     */
    public void chooseIterations(int iterations){
        this.iterations = iterations;
    }

    /**
     * For two examples, calculates the Euclidean Distance between them
     * @param e1
     * @param e2
     * @return Euclidean Distance
     */
    public double euclideanDist(Example e1, Example e2){
        double sum = 0;

        for(Integer feature : data.getAllFeatureIndices()){
            Double v1 = 0.0;
            Double v2 = 0.0;

            if(e1.getFeatureSet().contains(feature)){
                v1 = e1.getFeature(feature);
            }

            if(e2.getFeatureSet().contains(feature)){
                v2 = e2.getFeature(feature);
            }

            sum += Math.pow(v1 - v2, 2);
        }

        return Math.sqrt(sum);
    }

    /**
     * For two examples, calculates the Cosine similarity between them (range of 0 to 1)
     * @param e1
     * @param e2
     * @return Cosine Distance
     */
    public double cosineDistance(Example e1, Example e2){
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
        return nearestCentroid(example).getLabel();
    }

    @Override
    public double confidence(Example example) {
        double curDistance = 0;

        if(this.distChoice == COSINE_DIST){
            curDistance = cosineDistance(example, nearestCentroid(example));
        }else if(distChoice == EUCLIDEAN_DIST){
            curDistance = euclideanDist(example, nearestCentroid(example));
        }

        return curDistance;
    }
    
    /**
     * EVALUATION FUNCTION
     * For a labeled data and a given centroid, find the purity, aka
     * the proportion of the dominant class in the cluster
     * @param curCentroid
     * @return proportion of the biggest label in cluster
     */
    private double centroidPurity(Centroid curCentroid){
        HashMap<Double, Double> labelProportions = new HashMap<>();
        ArrayList<Example> curPoints = curCentroid.getAssociatedPoints();
        double majority = 0.0;

        for(int i=0;i<curPoints.size();i++){
            double curLabel = curPoints.get(i).getLabel();
            if(labelProportions.containsKey(curLabel)){
                labelProportions.put(curLabel, labelProportions.get(curLabel)+1.0);
            }else{
                labelProportions.put(curLabel, 1.0);
            }
        }

        for(Double curProportion : labelProportions.values()){
            if(curProportion > majority){
                majority = curProportion;
            }
        }

        return majority/curPoints.size();
    }

    /**
     * EVALUATION FUNCTION
     * For a given centroid and labeled data, evaluates entropy of the cluster
     * Entropy is a measure of disorder/randomness, lower is better here
     * Assumes labeled data
     * @param curCentroid
     * @return entropy for given centroid
     */
    private double centroidEntropy(Centroid curCentroid){
        HashMap<Double, Double> labelProportions = new HashMap<>();
        ArrayList<Example> curPoints = curCentroid.getAssociatedPoints();
        double entropy = 0.0;
        int size = curPoints.size();

        for(int i=0;i<curPoints.size();i++){
            double curLabel = curPoints.get(i).getLabel();
            if(labelProportions.containsKey(curLabel)){
                labelProportions.put(curLabel, labelProportions.get(curLabel)+1.0);
            }else{
                labelProportions.put(curLabel, 1.0);
            }
        }

        for(Double value : labelProportions.values()){
            entropy -= (value/size)*(Math.log(value/size));
        }

        return entropy;
    }

    /**
     * EVALUATION FUNCTION
     * For a particular centroid, calculate the distance of all points from the center, SSE
     * @param curCentroid
     * @return
     */
    private double centroidSSE(Centroid curCentroid){
        ArrayList<Example> curPoints = curCentroid.getAssociatedPoints();
        double sse = 0.0;

        for(int i=0; i<curPoints.size(); i++){
            if(distChoice == COSINE_DIST){
                sse += Math.pow(this.cosineDistance(curPoints.get(i), curCentroid), 2);
            }else if(distChoice == EUCLIDEAN_DIST){
                sse += Math.pow(this.euclideanDist(curPoints.get(i), curCentroid), 2);
            }
        }
        return sse;
    }

    /**
     * EVALUATION FUNCTION
     * Score between -1 (poorly defined cluster) and 1 (well defined cluster)
     * @param curExample
     * @return silhouette score for point
     */
    private double silhouetteScore(Centroid curCentroid, Example curExample){
        ArrayList<Example> curPoints = curCentroid.getAssociatedPoints();
        double intraDist = 0.0;
        double interDist = 0.0;

        for(int i=0;i<curPoints.size();i++){
            if(distChoice == EUCLIDEAN_DIST){
                intraDist += euclideanDist(curPoints.get(i), curExample);
            }else if (distChoice == COSINE_DIST){
                intraDist += cosineDistance(curPoints.get(i), curExample);
            }
        }

        Centroid nextNearest = nextNearestCentroid(curExample);
        ArrayList<Example> nextNearestPoints = nextNearest.getAssociatedPoints();

        for(int i=0;i<nextNearestPoints.size();i++){
            if(distChoice == EUCLIDEAN_DIST){
                interDist += euclideanDist(nextNearestPoints.get(i), curExample);
            }else if (distChoice == COSINE_DIST){
                interDist += cosineDistance(nextNearestPoints.get(i), curExample);
           }
        }
        
        //System.out.println(intraDist);
        intraDist /= curPoints.size();
        //System.out.println(interDist);
        interDist /= nextNearestPoints.size();
        //System.out.println(intraDist);
        //System.out.println(interDist);

        double res = (interDist - intraDist) / Math.max(intraDist, interDist);

        return res;
    }

    /**
     * EVALUATION FUNCTION
     * For a centroid, find the silhouette score across all all points in centroid
     * Score between -1 (poorly defined cluster) and 1 (well defined cluster)
     * @param curCentroid centroid in question
     * @return silhouette score for centroid
     */
    private double centroidSilhouette(Centroid curCentroid){
        ArrayList<Example> curPoints = curCentroid.getAssociatedPoints();
        double silhouetteScore = 0.0;

        for(int i=0;i<curPoints.size();i++){
            silhouetteScore += this.silhouetteScore(curCentroid, curPoints.get(i));
        }

        return silhouetteScore/curPoints.size();
    }

    /**
     * EVALUATION FUNCTION
     * Returns average model score for chosen evaluation criteria
     * @param scoreChoice Choice of evaluation Criteria
     */
    public double averageScore(int scoreChoice){
        double average = 0.0;
        for(int i=0;i<centroids.size();i++){
            if(scoreChoice == SILHOUETTE){
                // Silhouette Score
                average += centroidSilhouette(centroids.get(i));
            }else if(scoreChoice == SSE){
                // Sum of Squared Error
                average += centroidSSE(centroids.get(i));
            }else if(scoreChoice == ENTROPY){
                // Entropy
                average += centroidEntropy(centroids.get(i));
            }else if(scoreChoice == PURITY){
                // Purity
                average += centroidPurity(centroids.get(i));
            }
        }

        return average/centroids.size();
    }
}