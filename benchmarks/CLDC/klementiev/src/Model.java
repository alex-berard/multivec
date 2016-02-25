import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class Model {
    Map<Integer,SparseVector> class2Vector = new HashMap<Integer, SparseVector>();
    private Model() {
        
    }
    public Model(int maxLab, int maxFeatureId) {
        for (int i = 1; i <= maxLab; i++) {
            class2Vector.put(i, new SparseVector(maxFeatureId));
        }
    }
    
    public void addModel(Model m, double coeff) {
        for (Integer i : class2Vector.keySet()) {
            class2Vector.get(i).add(m.class2Vector.get(i), coeff);
        }
    }
    
    public static Model load(String fileName) throws IOException {
        BufferedReader in = new BufferedReader(new FileReader(fileName));
        
        Model mod = new Model();
        Example ex = null;
        while ((ex = Example.readExample(in))!= null) {
            mod.class2Vector.put(ex.label, ex.sv);
        }
        in.close();
        return mod;
        
    }
    
    public void save(String fileName) throws IOException {
        BufferedWriter out = new BufferedWriter(new FileWriter(fileName));
        for (Integer i : class2Vector.keySet()) {
            Example ex = new Example(i, class2Vector.get(i));
            ex.saveExample(out);
        }
        out.close();
    }
    
    public int predictLabel(SparseVector sv) {
        int optLab = -1;
        double opt = Double.NEGATIVE_INFINITY;
        for (int lab : class2Vector.keySet()) {
            double score = class2Vector.get(lab).innerProductForNonSparse(sv);
            if (score > opt) {
                optLab = lab;
                opt = score;
            }
        }
        return optLab;
    }
    
}
