package ml.classifiers;

import java.util.ArrayList;
import java.util.Random;

import ml.data.DataSet;
import ml.data.Example;

public class Experimenter {
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
    }
}
