package ml.data;
import java.util.*;

public class Centroid extends Example{
    ArrayList<Example> associatedExamples;

    public Centroid(ArrayList<Example> associatedExamples){
        this.associatedExamples = associatedExamples;
    }

    public void addExample(Example e){
        associatedExamples.add(e);
    }

    public void clearExamples(){
        associatedExamples.clear();
    }
}
