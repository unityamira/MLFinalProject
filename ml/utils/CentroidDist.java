package ml.utils;

import ml.data.Centroid;

public class CentroidDist implements Comparable<CentroidDist>{
    private Centroid centroid;
    private double distance;

    public CentroidDist(Centroid centroid, double distance){
        this.centroid = centroid;
        this.distance = distance;
    }

    public double getDistance(){ 
        return distance; 
    }

    public Centroid getCentroid(){ 
        return centroid; 
    }

    @Override
    public int compareTo(CentroidDist newCentroid) {
        return compare(this.distance, newCentroid.getDistance());
    }

    public static int compare(double dist1, double dist2){
        return (dist1 < dist2) ? -1: ((dist1==dist2) ? 0 :1);
    }
}