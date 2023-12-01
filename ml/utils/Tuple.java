package ml.utils;

import ml.data.Centroid;

public class CentroidDist implements Comparable<CentroidDist>{
    protected Centroid centroid;
    protected double label;

    public CentroidDist(Centroid centroid, double distance){
        this.distance = distance;
        this.label = label;
    }

    public double getDistance(){ return distance; }

    public double getLabel(){ return label; }

    @Override
    public int compareTo(DistExample dExample) {
        return compare(this.distance, dExample.getDistance());
    }

    public static int compare(double dist1, double dist2){
        return (dist1 < dist2) ? -1: ((dist1==dist2) ? 0 :1);
    }
}