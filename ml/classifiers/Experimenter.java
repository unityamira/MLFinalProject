package ml.classifiers;
import ml.data.*;

public class Experimenter {
    public static void main(String[] args){
        /*
        Random rand = new Random();

        Example e1 = simpleDataSet.getData().get(0);
        Example e2 = simpleDataSet.getData().get(1);
        Example e3 = simpleDataSet.getData().get(2);
        System.out.print("E1: ");
        System.out.println(e1);
        System.out.print("E2: ");
        System.out.println(e2);
        System.out.print("Cosine Distance:");
        System.out.println(model.cosineDistance(e1, e2));
        System.out.print("Euclidean Distance:");
        System.out.println(model.euclideanDist(e1, e2));
        System.out.println("");

        System.out.print("E1: ");
        System.out.println(e1);
        System.out.print("E2: ");
        System.out.println(e3);
        System.out.print("Cosine Distance:");
        System.out.println(model.cosineDistance(e1, e3));
        System.out.print("Euclidean Distance:");
        System.out.println(model.euclideanDist(e1, e3));


        for(int i=0;i<1;i++){
            Example ex1 = examples.get(rand.nextInt(wineDataSet.getData().size()));
            Example ex2 = examples.get(rand.nextInt(wineDataSet.getData().size()));

            System.out.println(ex1.toString(wineDataSet.getFeatureMap())+"\n");
            System.out.println(ex2.toString(wineDataSet.getFeatureMap())+"\n");

            System.out.print("Cosine Distance:");
            System.out.println(model.cosineDistance(ex1, ex2));
            System.out.print("Euclidean Distance:");
            System.out.println(model.euclideanDist(ex1, ex2));
        }
        */

        DataSet wineDataSet = new DataSet("data/wines.train", DataSet.TEXTFILE);
        //DataSet simpleDataSet = new DataSet("data/simple.csv", DataSet.CSVFILE);
        //ArrayList<Example> examples = wineDataSet.getData();

        /*
        for(int i=3;i<20;i++){
            KMeans model = new KMeans(i);
            model.chooseDistance(0);
            model.train(wineDataSet);
            System.out.println(model.averageScore(model.SILOUETTE));
        }
        */

        //DataSet simpleDataSet = new DataSet("data/test_data.csv", DataSet.CSVFILE);
        DataSet wineDataSet2 = new DataSet("data/wine.csv", DataSet.CSVFILE);
        //DataSet titanticDataSet = new DataSet("data/titanic-train.real.csv", DataSet.CSVFILE);
        

        for(int i=2;i<20;i++){
            double avg = 0.0;
            for(int j=0;j<10;j++){
                KMeans model = new KMeans(i);
                model.chooseInitialize(model.RANDOM_INIT);
                model.chooseDistance(model.COSINE_DIST);
                model.train(wineDataSet2);
                avg += model.averageScore(model.SILHOUETTE);
            }
            System.out.println(avg / 10);
        }
        
        
        /*
        DataSet simpleDataSet = new DataSet("data/test_data.csv", DataSet.CSVFILE);

        KMeans model = new KMeans(9);
        //model.chooseIterations(1);
        model.train(simpleDataSet);
        System.out.println(model.averageScore(3));
        */
    }
}
