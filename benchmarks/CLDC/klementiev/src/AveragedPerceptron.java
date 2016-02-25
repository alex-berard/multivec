import java.util.Collections;
import java.util.List;
import java.util.Random;


public class AveragedPerceptron {

    public AveragedPerceptron() {
        
    }
    
    private int getMaxFeatureId(List<Example> data) {
        int topId = -1;
        for (Example ex : data) {
            topId = ex.sv.topFeatureId() > topId ? ex.sv.topFeatureId() : topId; 
        }
        return topId;
    }
    

    private int getMaxLabelId(List<Example> data) {
        int topLabel = -1;
        for (Example ex : data) {
            int lab = ex.label;
            if (lab < 1) {
                throw new RuntimeException("Labels should be integers, above 0");
            }
            if (lab > topLabel) {
                topLabel = lab;
            }
            
        }
        return topLabel;
    }
    
    
    public Model learn(List<Example> data, int iterNum) {
        Collections.shuffle(data, new Random(0));
        
        Model mod = new Model(getMaxLabelId(data), getMaxFeatureId(data));
        Model avMod = new Model(getMaxLabelId(data), getMaxFeatureId(data));
        
        double coeff = 1. / (data.size() * iterNum);
        
        System.out.println("Learning");
        
        //int age = 0;
        int remaining = data.size() * iterNum;
        for (int e = 0; e < iterNum; e++) {
            int numUpd = 0;
            
            for (Example ex : data) {
                int pred = mod.predictLabel(ex.sv);
                if (pred != ex.label) {
                    mod.class2Vector.get(pred).add(ex.sv, - coeff);
                    mod.class2Vector.get(ex.label).add(ex.sv, coeff);
                    numUpd++;
                    //avMod.addModel(mod, age * 1);
                    avMod.class2Vector.get(pred).add(ex.sv, - coeff * remaining * coeff );
                    avMod.class2Vector.get(ex.label).add(ex.sv, coeff * remaining * coeff);
                    //age = 0;
                }
                //age++;
                remaining--;
            }
            System.out.print(numUpd + ".");
            
        }
        //avMod.addModel(mod, age * 1);
        
        System.out.println("Done");
        //return mod;
        return avMod;
        
    }
    
}
