package ml.data;
import java.util.*;

public class Centroid extends Example{
    ArrayList<Example> associatedExamples;

    public Centroid(){
        associatedExamples = new ArrayList<>();
    }

    public Centroid(ArrayList<Example> associatedExamples){
        this.associatedExamples = associatedExamples;
    }

    public void addExample(Example e){
        associatedExamples.add(e);
    }

    public void clearExamples(){
        associatedExamples.clear();
    }

    public ArrayList<Example> getAssociatedPoints(){
        return this.associatedExamples;
    }
}
